import pandas as pd
import os
import time
from datetime import datetime
import tempfile
from PIL import Image
import base64

# Import from the main script - fix import path using the same convention as streamlit app
from core.iris_analyzer_backend import (
    encode_image_to_base64, get_llava_response, is_generic_response,
    aggregate_marker_responses, MARKER_PROMPT_TEMPLATE
)

# --- Configuration ---
IRIDOLOGY_DATA_FILE = 'iridology-clean.xlsx'  # Path to cleaned Excel file
DEFAULT_RESIZE_WIDTH = 512
DEFAULT_RESIZE_HEIGHT = 512

def load_hierarchical_data(excel_file_path, sheet_name=0, priority_threshold=None, max_markers=None):
    """
    Loads the iridology data from Excel file into a hierarchical structure by body system.
    
    Args:
        excel_file_path: Path to Excel file containing iridology data
        sheet_name: Name or index of sheet to read
        priority_threshold: Maximum priority level to include (if None, include all)
        max_markers: Maximum number of markers to include per system (if None, include all)
    
    Returns:
        Dictionary organized by body system containing markers
    """
    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        print(f"Successfully read Excel file: {excel_file_path}, sheet: {sheet_name if isinstance(sheet_name, str) else 'first sheet'}")
        print(f"Available columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading data from Excel file '{excel_file_path}': {e}")
        return {}

    try:
        required_columns = ['Main Category', 'Sub-Category', 'Analysis(*)']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: The following required columns are missing from the file: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return {}
            
        # Make a copy of the DataFrame to avoid modifying the original
        df_cleaned = df[required_columns].copy()
        df_cleaned = df_cleaned.dropna(subset=['Analysis(*)'])
        
        # Check for Priority column
        has_priority = 'Priority' in df.columns
        if has_priority:
            # Add Priority column to df_cleaned from original df
            # First, make sure we're only adding values for rows that remain after dropna
            df_cleaned['Priority'] = df.loc[df_cleaned.index, 'Priority'].values
            print("Added Priority column to cleaned dataframe")
            
            # Filter by priority threshold if provided
            if priority_threshold is not None:
                df_cleaned = df_cleaned[df_cleaned['Priority'] <= priority_threshold]
                print(f"Filtered to {len(df_cleaned)} entries with priority <= {priority_threshold}")
        elif priority_threshold is not None:
            print(f"Warning: Priority column not found but priority_threshold={priority_threshold} was specified.")
            print(f"Will skip priority filtering and use all {len(df_cleaned)} markers.")
        
        # Check for Body System column
        has_body_system = 'Body System' in df.columns
        if not has_body_system:
            print("Warning: Body System column not found. Will assign body systems based on keywords in text.")
            
            # Create a simplified body system classification
            df_cleaned['Body System'] = df_cleaned.apply(
                lambda row: assign_body_system(row['Main Category'], row['Sub-Category'], row['Analysis(*)']), 
                axis=1
            )
            print("Assigned body systems based on text analysis")
        else:
            # Add Body System column to df_cleaned from original df (careful with indices)
            # We need the values only for the rows that remain in df_cleaned
            df_cleaned['Body System'] = df.loc[df_cleaned.index, 'Body System'].values
            print("Added Body System column to cleaned dataframe")
            
        # Create hierarchical structure grouped by body system
        hierarchical_data = {}
        
        # Get all unique body systems
        body_systems = df_cleaned['Body System'].unique()
        
        # For each body system, create a sub-dictionary with markers grouped by priority
        for system in body_systems:
            system_df = df_cleaned[df_cleaned['Body System'] == system]
            
            if has_priority:
                # Group by priority
                priorities = sorted(system_df['Priority'].unique())
                hierarchical_data[system] = {
                    int(priority): system_df[system_df['Priority'] == priority].to_dict('records')
                    for priority in priorities
                }
                
                # Apply max_markers per body system if specified
                if max_markers:
                    markers_count = 0
                    for priority in sorted(priorities):
                        if markers_count >= max_markers:
                            hierarchical_data[system][int(priority)] = []
                        else:
                            current_markers = len(hierarchical_data[system][int(priority)])
                            if markers_count + current_markers > max_markers:
                                # Trim to remain within max_markers
                                hierarchical_data[system][int(priority)] = hierarchical_data[system][int(priority)][:max_markers-markers_count]
                            markers_count += len(hierarchical_data[system][int(priority)])
            else:
                # If no priority column, just group by body system
                system_records = system_df.to_dict('records')
                if max_markers and len(system_records) > max_markers:
                    system_records = system_records[:max_markers]
                hierarchical_data[system] = {"all": system_records}
        
        # Count total markers for logging
        total_markers = 0
        for system, priorities in hierarchical_data.items():
            for priority, markers in priorities.items():
                total_markers += len(markers)
        
        print(f"Successfully processed {total_markers} iridology signs organized by body system.")
        return hierarchical_data
            
    except Exception as e:
        print(f"Error processing dataframe for hierarchical structure: {e}")
        import traceback
        traceback.print_exc()
        return {}

def assign_body_system(main_category, sub_category, analysis_text):
    """
    Assign a body system based on the text in the main category, sub category, and analysis text.
    
    Args:
        main_category: Main category from the Excel file
        sub_category: Sub category from the Excel file
        analysis_text: Analysis text from the Excel file
        
    Returns:
        String with the assigned body system
    """
    # Define keywords for each body system
    body_systems = {
        "Digestive": ["digest", "stomach", "intestine", "bowel", "colon", "gallbladder", 
                     "liver", "pancreas", "acid", "digestion", "absorption", "reflux"],
        "Nervous": ["nervous", "stress", "anxiety", "tension", "mental", "pineal", "melatonin", 
                   "sleep", "nerve", "brain", "headache", "emotional"],
        "Endocrine": ["hormone", "thyroid", "adrenal", "pituitary", "insulin", "pancreas", 
                     "blood sugar", "metabolism", "energy"],
        "Circulatory": ["circulatory", "heart", "blood", "circulation", "cardiovascular", 
                        "cholesterol", "artery", "vein", "oxygen", "iron"],
        "Lymphatic": ["lymph", "lymphatic", "immune", "tonsil", "spleen", "toxin", "drainage",
                     "congestion", "inflammation"],
        "Respiratory": ["respiratory", "lung", "breathing", "breath", "chest", "air", "respiration"],
        "Structural": ["structural", "spine", "neck", "joint", "knee", "posture", "alignment", 
                      "muscle", "bone", "back"],
        "Urinary": ["urinary", "kidney", "bladder", "urine", "water", "fluid"]
    }
    
    # Combine all text fields for analysis, make lowercase
    all_text = f"{main_category} {sub_category} {analysis_text}".lower()
    
    # Direct category mapping (if main category directly indicates a system)
    category_to_system = {
        'digestive': 'Digestive',
        'nervous system': 'Nervous', 
        'nervous': 'Nervous',
        'endocrine': 'Endocrine',
        'circulatory': 'Circulatory', 
        'circulation': 'Circulatory',
        'lymphatic': 'Lymphatic',
        'respiratory': 'Respiratory', 
        'respiration': 'Respiratory',
        'structural': 'Structural',
        'urinary': 'Urinary'
    }
    
    # Check if main category directly maps to a system
    main_category_lower = main_category.lower()
    if main_category_lower in category_to_system:
        return category_to_system[main_category_lower]
    
    # Find matching keywords for each system
    system_matches = {}
    for system, keywords in body_systems.items():
        match_count = sum(1 for keyword in keywords if keyword in all_text)
        if match_count > 0:
            system_matches[system] = match_count
    
    # Return the system with the most matches, or "Other" if no matches
    if system_matches:
        return max(system_matches.items(), key=lambda x: x[1])[0]
    else:
        return "Other"

def analyze_iris_image_hierarchical(image_path_or_bytes, hierarchical_data, max_markers_per_system=3, early_stop_threshold=2):
    """
    Analyzes an iris image using a hierarchical approach, stopping early when findings are detected.
    
    Args:
        image_path_or_bytes: Path to image file or bytes of the image
        hierarchical_data: Hierarchical dictionary of iridology markers organized by body system and priority
        max_markers_per_system: Maximum number of markers to check per body system
        early_stop_threshold: Number of positive findings needed to stop checking a body system
        
    Returns:
        List of dictionaries with marker responses
    """
    if not hierarchical_data:
        return "Error: Hierarchical iridology data is empty. Cannot perform analysis."

    image_base64 = encode_image_to_base64(image_path_or_bytes)
    if not image_base64:
        return "Error: Could not encode image."
    
    # Keep track of all marker responses
    all_marker_responses = []
    # Keep track of positive findings per system
    system_findings = {}
    
    # Analysis order: we'll process systems in this order for a logical flow
    analysis_order = [
        "Digestive", "Circulatory", "Nervous", "Endocrine", 
        "Lymphatic", "Respiratory", "Urinary", "Structural", "Other"
    ]
    
    # Make sure all systems in hierarchical_data are included
    for system in hierarchical_data.keys():
        if system not in analysis_order:
            analysis_order.append(system)
    
    print(f"\n--- Starting Hierarchical Analysis for Image ---")
    print(f"Strategy: Analyze up to {max_markers_per_system} markers per body system, "
          f"stop early after finding {early_stop_threshold} positive markers")
    
    # Process each body system in the specified order
    for system in analysis_order:
        if system not in hierarchical_data:
            continue
            
        system_markers = []
        system_positive_count = 0
        
        print(f"\n--- Analyzing {system} System ---")
        
        # Process markers by priority (1, 2, 3, or 'all' if no priority)
        priorities = sorted(hierarchical_data[system].keys()) if isinstance(hierarchical_data[system], dict) else ['all']
        
        for priority in priorities:
            # Get markers for this priority level (handle both dict and list formats)
            markers = hierarchical_data[system][priority] if isinstance(hierarchical_data[system], dict) else hierarchical_data[system]
            
            # Skip if no markers in this priority group
            if not markers:
                continue
                
            # Keep track of how many markers we've analyzed for this system
            markers_analyzed = 0
            
            # Process each marker in order of priority
            for i, marker_info in enumerate(markers):
                # Skip if we've reached our limit for this system
                if markers_analyzed >= max_markers_per_system:
                    break
                    
                # Skip if we've already found enough positive markers in this system
                if system_positive_count >= early_stop_threshold:
                    print(f"Found {system_positive_count} positive markers in {system} system. "
                          f"Stopping analysis for this system.")
                    break
                
                marker_name = marker_info.get('Analysis(*)', 'No specific marker listed.')
                main_category = marker_info.get('Main Category', '')
                sub_category = marker_info.get('Sub-Category', '')
                
                if not marker_name or marker_name.strip() == "":
                    continue
                
                # Format the marker for analysis
                full_marker_name = marker_name
                if main_category and sub_category:
                    # Keep it shorter to save tokens
                    full_marker_name = f"{marker_name} [{main_category}: {sub_category}]"
                
                # Format the prompt with the marker name
                marker_prompt = MARKER_PROMPT_TEMPLATE.format(marker_name=full_marker_name)
                
                print(f"\n[{system} Marker {i+1}] Analyzing: {full_marker_name[:70]}...")
                
                # Analyze this marker
                response = get_llava_response(image_base64, marker_prompt, current_max_tokens=200)
                
                # Check if response is generic
                is_generic = is_generic_response(response)
                if is_generic:
                    print(f"⚠️ WARNING: Response may be generic or uncertain")
                    # Print a shorter preview of the response
                    preview = response.replace("\n", " ")[:100] + "..." if len(response) > 100 else response
                    print(f"LLaVA's Response: {preview}")
                else:
                    # Print a shorter preview of the response
                    preview = response.replace("\n", " ")[:100] + "..." if len(response) > 100 else response
                    print(f"LLaVA's Response: {preview}")
                
                # Check if this is a positive finding
                is_positive = False
                if "KEY FINDING: YES" in response.upper() or "KEY FINDING: POSSIBLE" in response.upper():
                    is_positive = True
                    system_positive_count += 1
                    print(f"✅ POSITIVE FINDING detected in {system} system")
                
                # Store the marker response
                marker_response = {
                    "marker": full_marker_name,
                    "main_category": main_category,
                    "sub_category": sub_category,
                    "body_system": system,
                    "response": response,
                    "is_generic": is_generic,
                    "is_positive": is_positive
                }
                
                system_markers.append(marker_response)
                all_marker_responses.append(marker_response)
                markers_analyzed += 1
        
        # Store the findings for this system
        system_findings[system] = {
            "markers_analyzed": len(system_markers),
            "positive_findings": system_positive_count,
            "marker_responses": system_markers
        }
        
        print(f"Completed analysis of {system} system: "
              f"Analyzed {len(system_markers)} markers, "
              f"Found {system_positive_count} positive findings")
    
    # Log summary of findings by system
    print("\n--- Hierarchical Analysis Results by Body System ---")
    total_markers_analyzed = 0
    total_positive_findings = 0
    
    for system, findings in system_findings.items():
        markers_analyzed = findings["markers_analyzed"]
        positive_findings = findings["positive_findings"]
        total_markers_analyzed += markers_analyzed
        total_positive_findings += positive_findings
        
        print(f"  • {system}: {positive_findings}/{markers_analyzed} positive findings")
    
    print(f"Total: {total_positive_findings}/{total_markers_analyzed} positive findings")
    print("----------------------------------------")
    
    return all_marker_responses

def resize_image_before_processing(image_path, target_size=(DEFAULT_RESIZE_WIDTH, DEFAULT_RESIZE_HEIGHT)):
    """
    Resize image to reduce token count during encoding.
    
    Args:
        image_path: Path to the original image file
        target_size: Tuple of (width, height) for the resized image
        
    Returns:
        Path to the temporary resized image file
    """
    try:
        # Open the image
        img = Image.open(image_path)
        original_size = img.size
        
        # Resize the image
        resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Save to temporary file
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        resized.save(temp.name, format="JPEG", quality=85)
        
        print(f"Image resized from {original_size} to {target_size} to reduce token count")
        return temp.name
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_path  # Return original path on error

def enhanced_aggregate_marker_responses(marker_responses):
    """
    Aggregates and formats marker responses into a concise report with enhanced recommendations.
    This improved version organizes findings by body system and confidence level.
    
    Args:
        marker_responses (list): List of dictionary responses from marker analysis
        
    Returns:
        str: Formatted report with recommendations based on findings
    """
    # Debug information
    print(f"Processing {len(marker_responses)} marker responses for report generation")
    
    # Organize findings by body system and confidence
    body_system_findings = {}
    
    # Initialize counters for overall statistics
    abnormal_count = 0
    possible_count = 0
    normal_count = 0
    uncertain_count = 0
    
    # Process each marker response
    for resp in marker_responses:
        marker_name = resp["marker"]
        response = resp["response"]
        body_system = resp.get("body_system", "General")
        
        print(f"Processing marker: {marker_name[:30]}... in {body_system} system")
        
        # Initialize the body system entry if it doesn't exist
        if body_system not in body_system_findings:
            body_system_findings[body_system] = {
                "definitive": [],
                "possible": [], 
                "normal": [],
                "uncertain": []
            }
        
        # Extract the key finding and confidence using our structured format
        key_finding = "No clear finding"
        confidence = "Uncertain"
        is_abnormal = False
        is_uncertain = False
        is_possible = False
        is_generic = resp.get("is_generic", False)  # Default to False if not provided
        
        # Try to extract structured data
        if "KEY FINDING:" in response:
            key_finding_section = response.split("KEY FINDING:")[1].split("\n")[0].strip()
            key_finding = key_finding_section
            print(f"  Key finding: {key_finding}")
            
        if "CONFIDENCE:" in response:
            confidence_section = response.split("CONFIDENCE:")[1].split("\n")[0].strip()
            confidence = confidence_section
            
        # Extract visual details for better reporting
        visual_details = "No detailed description available"
        if "VISUAL DETAILS:" in response:
            visual_details_section = response.split("VISUAL DETAILS:")[1]
            # Get everything until the next major section or the end
            if "SELF-VERIFICATION:" in visual_details_section:
                visual_details = visual_details_section.split("SELF-VERIFICATION:")[0].strip()
            else:
                visual_details = visual_details_section.split("\n\n")[0].strip()
                
        # Extract self-verification
        self_verified = "No"
        if "SELF-VERIFICATION:" in response:
            self_verified = response.split("SELF-VERIFICATION:")[1].split("\n")[0].strip()
            
        # Check the finding type based on key words and the is_positive field if available
        key_finding_lower = key_finding.lower()
        
        # Use is_positive from marker if available (set by hierarchical analysis)
        if "is_positive" in resp and resp["is_positive"]:
            print(f"  Marker marked as positive by hierarchical analysis")
            if "yes" in key_finding_lower:
                print(f"  Identified as definitive abnormal finding")
                is_abnormal = True
                abnormal_count += 1
            else:
                print(f"  Identified as possible abnormal finding")
                is_possible = True
                possible_count += 1
        elif "not observed" in key_finding_lower or "no " in key_finding_lower or "absent" in key_finding_lower:
            is_abnormal = False
            normal_count += 1
            print(f"  Identified as normal finding")
        elif "possible" in key_finding_lower or "may be" in key_finding_lower:
            is_possible = True
            possible_count += 1
            print(f"  Identified as possible abnormal finding")
        elif "yes" in key_finding_lower or "present" in key_finding_lower or "observed" in key_finding_lower:
            is_abnormal = True
            abnormal_count += 1
            print(f"  Identified as definitive abnormal finding")
        else:
            # Only mark as uncertain if no clear classification
            is_uncertain = (key_finding == "No clear finding")
            if is_uncertain:
                uncertain_count += 1
                print(f"  Identified as uncertain finding (no clear key finding)")
            else:
                # Default to normal if not clearly classified
                normal_count += 1
                print(f"  Defaulting to normal finding")
            
        # If self-verification is No or confidence is Low, don't mark as uncertain,
        # but note it in the report entry
        low_confidence = (self_verified.lower() == "no" or confidence.lower() == "low")
        
        # Create a formatted entry
        entry = {
            "marker": marker_name,
            "key_finding": key_finding,
            "confidence": confidence,
            "visual_details": visual_details,
            "self_verified": self_verified,
            "full_response": response,
            "is_abnormal": is_abnormal,
            "is_possible": is_possible,
            "is_uncertain": is_uncertain,
            "is_generic": is_generic,
            "low_confidence": low_confidence
        }
        
        # Add to the appropriate category in the body system
        category = "uncertain"
        if is_uncertain:
            category = "uncertain"
        elif is_possible:
            category = "possible"
        elif is_abnormal:
            category = "definitive"
        else:
            category = "normal"
            
        print(f"  Categorizing as: {category}")
        body_system_findings[body_system][category].append(entry)
    
    # Print debug info about findings by system
    print("\nSummary of findings by body system:")
    for system, categories in body_system_findings.items():
        print(f"  {system} System:")
        print(f"    Definitive: {len(categories['definitive'])}")
        print(f"    Possible: {len(categories['possible'])}")
        print(f"    Normal: {len(categories['normal'])}")
        print(f"    Uncertain: {len(categories['uncertain'])}")
    
    # Start building the report
    report = "ENHANCED IRIDOLOGY ANALYSIS SUMMARY\n"
    report += "===================================\n\n"
    
    # Add quick statistics
    report += f"STATISTICS:\n"
    report += f"• Definitively Abnormal: {abnormal_count} markers\n"
    report += f"• Possibly Abnormal: {possible_count} markers\n"
    report += f"• Normal/Absent: {normal_count} markers\n"
    report += f"• Uncertain/Flagged: {uncertain_count} markers\n\n"
    
    # Calculate system scores to sort by significance
    system_scores = {}
    for system, categories in body_system_findings.items():
        # Weight: definitive (3), possible (2), normal (-1), uncertain (0)
        score = len(categories["definitive"]) * 3 + len(categories["possible"]) * 2 - len(categories["normal"])
        system_scores[system] = score
    
    # Sort systems by their significance score
    sorted_systems = sorted(system_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Add system-by-system findings
    report += "KEY FINDINGS BY BODY SYSTEM\n"
    report += "===========================\n\n"
    
    for system, score in sorted_systems:
        categories = body_system_findings[system]
        if score <= 0 and not (categories["definitive"] or categories["possible"]):
            continue  # Skip systems with no significant findings
        
        report += f"## {system.upper()} SYSTEM\n"
        
        # Add definitive findings
        if categories["definitive"]:
            report += "Definitive Findings:\n"
            for finding in categories["definitive"]:
                marker_name = finding['marker'].split('[')[0].strip()
                report += f"✅ {marker_name}\n"
                report += f"   Confidence: {finding['confidence']}\n"
                if finding['visual_details'] != "No detailed description available":
                    details = finding['visual_details'].replace('\n', ' ').strip()
                    report += f"   Details: {details[:100]}...\n" if len(details) > 100 else f"   Details: {details}\n"
            report += "\n"
        
        # Add possible findings
        if categories["possible"]:
            report += "Possible Findings:\n"
            for finding in categories["possible"]:
                marker_name = finding['marker'].split('[')[0].strip()
                report += f"⚠️ {marker_name}\n"
                report += f"   Confidence: {finding['confidence']}\n"
            report += "\n"
        
        # Skip normal findings to keep report concise
        # Skip uncertain findings to focus on actionable items
        
    # Generate recommendations based on findings
    report += "RECOMMENDATIONS\n"
    report += "===============\n\n"
    
    # Group recommendations by priority (based on system score and finding confidence)
    priority_recommendations = []
    secondary_recommendations = []
    
    print("\nGenerating recommendations for systems:")
    for system, score in sorted_systems:
        print(f"  {system} system (score: {score})")
        if score <= 0:
            print(f"    Skipping - score too low")
            continue
            
        categories = body_system_findings[system]
        
        print(f"    Has definitive findings: {bool(categories['definitive'])}")
        print(f"    Has possible findings: {bool(categories['possible'])}")
        
        # Generate system-specific recommendations
        system_recs = generate_system_recommendations(system, categories)
        
        print(f"    Generated {len(system_recs)} recommendations")
        
        if score > 3 and categories["definitive"]:
            priority_recommendations.extend(system_recs)
        else:
            secondary_recommendations.extend(system_recs)
    
    # Add priority recommendations
    if priority_recommendations:
        report += "Primary Focus Areas:\n"
        for i, rec in enumerate(priority_recommendations[:3], 1):  # Limit to top 3
            report += f"{i}. {rec}\n"
        report += "\n"
    
    # Add secondary recommendations
    if secondary_recommendations:
        report += "Secondary Support:\n"
        for i, rec in enumerate(secondary_recommendations[:3], 1):  # Limit to top 3
            report += f"{i}. {rec}\n"
        report += "\n"
    
    # Add lifestyle recommendations based on overall patterns
    nervous_issues = (
        body_system_findings.get("Nervous", {}).get("definitive", []) or 
        body_system_findings.get("Nervous", {}).get("possible", [])
    )
    
    digestive_issues = (
        body_system_findings.get("Digestive", {}).get("definitive", []) or 
        body_system_findings.get("Digestive", {}).get("possible", [])
    )
    
    circulatory_issues = (
        body_system_findings.get("Circulatory", {}).get("definitive", []) or 
        body_system_findings.get("Circulatory", {}).get("possible", [])
    )
    
    report += "Lifestyle Recommendations:\n"
    
    if nervous_issues:
        report += "• Stress management: Regular deep breathing exercises, meditation, and adequate rest\n"
        report += "• Consider nervine herbs like chamomile, lemon balm, or passionflower\n"
    
    if digestive_issues:
        report += "• Dietary adjustments: Increase fiber intake, chew thoroughly, stay hydrated\n"
        report += "• Consider digestive support herbs like ginger, peppermint, or fennel\n"
    
    if circulatory_issues:
        report += "• Improve circulation: Regular moderate exercise, reduce inflammatory foods\n"
        report += "• Consider circulatory support herbs like hawthorn, ginkgo, or garlic\n"
    
    # Add general recommendations if no specific issues were found
    if not (nervous_issues or digestive_issues or circulatory_issues):
        report += "• General wellness: Stay hydrated, maintain regular physical activity\n"
        report += "• Consider a whole-foods diet rich in vegetables and anti-inflammatory foods\n"
    
    report += "\nNote: This analysis is based on iridology principles and is not a medical diagnosis.\n"
    report += "Consult with a qualified healthcare practitioner for personalized advice.\n"
    
    return report

def generate_system_recommendations(system, categories):
    """Generate system-specific recommendations based on findings"""
    recommendations = []
    
    # Check if this system has any findings to report
    has_findings = bool(categories.get("definitive") or categories.get("possible"))
    
    # System-specific recommendations based on findings
    if system == "Nervous":
        if categories.get("definitive"):
            recommendations.append("Support nervous system through stress reduction techniques, deep relaxation practices, and adaptogenic herbs")
        if any("stress" in finding.get("marker", "").lower() for finding in categories.get("definitive", []) + categories.get("possible", [])):
            recommendations.append("Focus on nervous system support through deep breathing, meditation, and nervous system tonics")
    
    elif system == "Digestive":
        if has_findings:
            recommendations.append("Improve digestive function with dietary modifications, digestive enzymes, and gut-healing nutrients")
        if any("pancrea" in finding.get("marker", "").lower() for finding in categories.get("definitive", []) + categories.get("possible", [])):
            recommendations.append("Support blood sugar balance with chromium, cinnamon, and regular balanced meals")
    
    elif system == "Circulatory":
        if has_findings:
            recommendations.append("Enhance circulation with cardiovascular exercise, circulation-supporting herbs, and anti-inflammatory diet")
        if any("cholesterol" in finding.get("marker", "").lower() for finding in categories.get("definitive", []) + categories.get("possible", [])):
            recommendations.append("Address cardiovascular health with omega-3 fatty acids, garlic, and cholesterol-balancing foods")
    
    elif system == "Lymphatic":
        if has_findings:
            recommendations.append("Support lymphatic drainage through movement, dry brushing, and lymphatic-stimulating herbs")
        if any("congest" in finding.get("marker", "").lower() for finding in categories.get("definitive", []) + categories.get("possible", [])):
            recommendations.append("Improve lymphatic flow with rebounding exercise, hydration, and lymphatic herbs like red root or cleavers")
    
    elif system == "Respiratory":
        if has_findings:
            recommendations.append("Support respiratory function with deep breathing exercises, steam inhalation, and respiratory herbs")
    
    elif system == "Endocrine":
        if has_findings:
            recommendations.append("Balance hormonal function with adaptogenic herbs, essential fatty acids, and targeted nutrition")
        if any("thyroid" in finding.get("marker", "").lower() for finding in categories.get("definitive", []) + categories.get("possible", [])):
            recommendations.append("Support thyroid health with selenium, iodine (if appropriate), and adaptogenic herbs")
    
    elif system == "Urinary":
        if has_findings:
            recommendations.append("Support kidney and bladder health with adequate hydration, urinary tract herbs, and reduced irritants")
    
    elif system == "Structural":
        if has_findings:
            recommendations.append("Address structural alignment through posture work, targeted exercise, and anti-inflammatory support")
    
    # If no specific recommendations were generated, add a generic one
    if not recommendations and has_findings:
        recommendations.append(f"Address {system} system issues with targeted nutrition, appropriate movement, and herbal support")
    
    # Debug print
    if has_findings and not recommendations:
        print(f"WARNING: System {system} has findings but no recommendations were generated")
    
    return recommendations

# Main execution when run as a script
if __name__ == "__main__":
    import sys
    
    # Use a specific test image - could be a command line arg
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = 'images/16.jpg'
        
    if not os.path.exists(test_image_path):
        print(f"Error: Test image '{test_image_path}' not found.")
        sys.exit(1)
        
    print(f"Using test image: {test_image_path}")
    
    # Resize the image to reduce token count
    original_image_path = test_image_path
    test_image_path = resize_image_before_processing(test_image_path)
    
    # Load hierarchical data
    print("\n--- Loading hierarchical iridology data ---")
    hierarchical_data = load_hierarchical_data(
        IRIDOLOGY_DATA_FILE,
        priority_threshold=2,  # Include priority 1-2 (high and medium)
        max_markers=5          # Up to 5 markers per body system
    )
    
    # Analyze using hierarchical approach
    marker_responses = analyze_iris_image_hierarchical(
        test_image_path, 
        hierarchical_data,
        max_markers_per_system=3,  # Analyze up to 3 markers per system
        early_stop_threshold=2     # Stop after finding 2 positive markers in a system
    )
    
    # Generate final report
    print("\n--- Creating Final Report ---")
    try:
        # Instead of directly importing aggregate_marker_responses from the main script,
        # use our enhanced version that provides better recommendations
        # from iris_analyzer_backend import aggregate_marker_responses
        final_report = enhanced_aggregate_marker_responses(marker_responses)
        
        print("\n\n--- FINAL IRIDOLOGY ANALYSIS (HIERARCHICAL) ---")
        print(final_report)
        
        # Append results to file
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, "iris_16_analysis.txt")
        
        with open(output_file_path, "a") as f:
            f.write(f"\n\n--- Enhanced Hierarchical Analysis from iris_analyze_hierarchical.py ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
            f.write(f"Image: {original_image_path} (resized to {DEFAULT_RESIZE_WIDTH}x{DEFAULT_RESIZE_HEIGHT})\n\n")
            f.write(final_report)
            
        print(f"\nResults appended to: {output_file_path}")
    except Exception as e:
        print(f"Error generating final report: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up the temporary resized image
    if test_image_path != original_image_path and os.path.exists(test_image_path):
        try:
            os.remove(test_image_path)
            print(f"Cleaned up temporary resized image: {test_image_path}")
        except Exception as e:
            print(f"Warning: Could not delete temporary file {test_image_path}: {e}") 