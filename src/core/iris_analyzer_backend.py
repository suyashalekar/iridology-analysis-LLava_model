import pandas as pd
# import requests # No longer needed directly here
import base64 # For encoding images
import io # For handling image data in memory
# import json # No longer needed directly here
import os
import time
from datetime import datetime
import tempfile # For temporary image file
from PIL import Image  # For image resizing

# Import from llava_mlx_client
from models.llava_mlx_client import query_llava as client_query_llava

# --- Configuration ---
# Path to your expert iridology data CSV
IRIDOLOGY_DATA_FILE = 'iridology-clean.xlsx' # Path to cleaned Excel file
LM_STUDIO_API_URL = 'http://localhost:1234/v1/chat/completions' # Default LM Studio API endpoint
MODEL_NAME = "llava-v1.6-vicuna-13b" # UPDATED MODEL NAME
# Default image size for resizing to reduce token count
DEFAULT_RESIZE_WIDTH = 512
DEFAULT_RESIZE_HEIGHT = 512

# --- Image Preprocessing ---
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

# --- 1. Load Expert Iridology Knowledge ---
def load_iridology_data(excel_file_path, sheet_name=0, filter_categories=None, priority_threshold=None, max_markers=None, hierarchical=False):
    """
    Loads the iridology checklist from the provided Excel file with filtering options.
    
    Args:
        excel_file_path: Path to Excel file containing iridology data
        sheet_name: Name or index of sheet to read
        filter_categories: List of main categories to include (if None, include all)
        priority_threshold: Maximum priority level to include (if None, include all)
        max_markers: Maximum number of markers to include (if None, include all)
        hierarchical: If True, return data in hierarchical structure by body system
    
    Returns:
        If hierarchical=True: Dictionary organized by body system
        If hierarchical=False: List of dictionaries containing filtered iridology data
    """
    try:
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
        print(f"Successfully read Excel file: {excel_file_path}, sheet: {sheet_name if isinstance(sheet_name, str) else 'first sheet'}")
        print(f"Available columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading iridology data from Excel file '{excel_file_path}': {e}")
        print("Trying to load from CSV fallback...")
        csv_file_path = 'iridology-checklist.csv'
        try:
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                print(f"Successfully read CSV file: {csv_file_path}")
            else:
                print(f"Error: CSV fallback file '{csv_file_path}' was not found.")
                return [] if not hierarchical else {}
        except Exception as csv_e:
            print(f"Error loading CSV fallback: {csv_e}")
        return [] if not hierarchical else {}

    try:
        required_columns = ['Main Category', 'Sub-Category', 'Analysis(*)']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: The following required columns are missing from the file: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return [] if not hierarchical else {}
            
        df_cleaned = df[required_columns].dropna(subset=['Analysis(*)'])
        
        # Apply category filter if provided
        if filter_categories:
            df_cleaned = df_cleaned[df_cleaned['Main Category'].isin(filter_categories)]
            print(f"Filtered to {len(df_cleaned)} entries based on categories: {filter_categories}")
        
        # Check for Priority column in the original dataframe
        has_priority = 'Priority' in df.columns
        if has_priority:
            # Add Priority column to df_cleaned from original df
            df_cleaned['Priority'] = df['Priority'].values
            print("Added Priority column to cleaned dataframe")
            
            # Sort by Priority 
            df_cleaned = df_cleaned.sort_values('Priority')
            print(f"Sorted markers by priority")
            
            # Filter by priority threshold if provided
            if priority_threshold is not None:
                df_cleaned = df_cleaned[df_cleaned['Priority'] <= priority_threshold]
                print(f"Filtered to {len(df_cleaned)} entries with priority <= {priority_threshold}")
        elif priority_threshold is not None:
            print(f"Warning: Priority column not found but priority_threshold={priority_threshold} was specified.")
            print(f"Will skip priority filtering and use all {len(df_cleaned)} markers.")
        
        # Check for Body System column
        has_body_system = 'Body System' in df.columns
        if not has_body_system and hierarchical:
            print("Warning: Body System column not found but hierarchical=True was specified.")
            print("Will assign body systems based on keywords in text.")
            
            # Create a simplified body system classification function
            df_cleaned['Body System'] = df_cleaned.apply(
                lambda row: assign_body_system(row['Main Category'], row['Sub-Category'], row['Analysis(*)']), 
                axis=1
            )
            print("Assigned body systems based on text analysis")
        elif has_body_system:
            # Add Body System column to df_cleaned from original df
            df_cleaned['Body System'] = df['Body System'].values
            print("Added Body System column to cleaned dataframe")
            
        # Apply maximum markers limit if provided (before hierarchical structuring)
        if max_markers and max_markers < len(df_cleaned) and not hierarchical:
            df_cleaned = df_cleaned.head(max_markers)
            print(f"Limited to top {max_markers} markers")
        
        # Create a standard flat list or hierarchical structure based on the parameter
        if not hierarchical:
            iridology_checklist = df_cleaned.to_dict('records')
            print(f"Successfully processed {len(iridology_checklist)} iridology signs to check.")
            return iridology_checklist
        else:
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
                        priority: system_df[system_df['Priority'] == priority].to_dict('records')
                        for priority in priorities
                    }
                    
                    # Apply max_markers per body system if specified
                    if max_markers:
                        markers_count = 0
                        for priority in priorities:
                            if markers_count >= max_markers:
                                hierarchical_data[system][priority] = []
                            else:
                                current_markers = len(hierarchical_data[system][priority])
                                if markers_count + current_markers > max_markers:
                                    # Trim to remain within max_markers
                                    hierarchical_data[system][priority] = hierarchical_data[system][priority][:max_markers-markers_count]
                                markers_count += len(hierarchical_data[system][priority])
                else:
                    # If no priority column, just group by body system
                    system_records = system_df.to_dict('records')
                    if max_markers and len(system_records) > max_markers:
                        system_records = system_records[:max_markers]
                    hierarchical_data[system] = {"all": system_records}
            
            # Count total markers for logging
            total_markers = sum(
                len(markers) for system in hierarchical_data.values() 
                for priority_group in system.values() 
                for markers in ([priority_group] if isinstance(priority_group, list) else [])
            )
            
            print(f"Successfully processed {total_markers} iridology signs organized by body system.")
            return hierarchical_data
            
    except KeyError as e:
        print(f"Error: One of the required columns is missing. Error: {e}")
        return [] if not hierarchical else {}
    except Exception as e:
        print(f"Error processing dataframe after loading: {e}")
        return [] if not hierarchical else {}

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

# --- 2. Define Master Prompt Templates ---
SYSTEM_PROMPT = "You are an AI Iridology Assistant."

PROBING_PROMPT_TEMPLATE = """
Analyze the provided iris image for signs related to '{sub_category}' within the '{main_category}'.
Expert guidance for '{sub_category}' indicates looking for: '{detailed_markers}'.

Based on the visual evidence in the image and the expert guidance:
For each distinct marker or sign concept implied by the expert guidance for '{sub_category}':

If OBSERVED:
* State 'OBSERVED: [Specific Marker/Sign]'
* Provide a detailed visual description including:
  - Location: [Specific area in the iris - e.g., "outer third", "6 o'clock position"]
  - Color: [Exact color and intensity - e.g., "bright white", "faint yellow"]
  - Shape: [Precise shape and pattern - e.g., "concentric circle", "radial lines"]
  - Size: [Relative size and extent - e.g., "covers 30% of the iris", "2mm in diameter"]
  - Clarity: [How distinct or clear the sign is - e.g., "well-defined", "fuzzy edges"]
  - Confidence Level: [High/Medium/Low] with brief explanation
* Example: 'OBSERVED: Stress ring - Located in the outer third of the iris, appears as a distinct white concentric circle (2mm width) with clear definition. High confidence due to clear visibility and textbook appearance.'

If NOT OBSERVED:
* State 'NOT OBSERVED: [Specific Marker/Sign]'
* Provide context:
  - Area checked: [Specific area examined]
  - Reason for non-observation: [Brief explanation if relevant]
  - Alternative observations: [Any related signs that were seen instead]
* Example: 'NOT OBSERVED: Cholesterol ring - Thoroughly examined the periphery of the iris (outer 2mm) with no visible white ring formation. The area shows normal iris texture and coloration.'

Additional Notes:
* Comment on image quality if it affects observation
* Note any areas that are unclear or need better lighting
* Mention if the sign might be partially visible or ambiguous

Focus on specific visual details and be precise in your descriptions. Avoid general statements about eye color unless it's specifically relevant to a marker.
"""

# NEW: Prompt for Stage 1 Synthesis (Summarizing Probes)
PROBE_SUMMARY_PROMPT_TEMPLATE = """
The following are a series of observations made about a single iris image.
IMPORTANT: I need you to KEEP ALL significant observations, especially markers described as 'OBSERVED'.

--- BEGIN OBSERVATIONS LIST ---
{all_raw_probe_observations}
--- END OBSERVATIONS LIST ---

Your task is to extract and compile ONLY the POSITIVE findings from the observations list:

1. Extract EVERY mention of an observed marker, even if it appears multiple times
2. Include ANY observation that:
   - Contains 'OBSERVED:' or similar positive wording
   - Describes a visible feature, pattern, or abnormality
   - Contains specific details about location, color, or appearance
   - Makes a definitive statement about what is seen in the iris

3. Format each finding as a clear bullet point with the SPECIFIC NAME of the marker and its DETAILED description
4. IMPORTANT: If the same marker appears multiple times, INCLUDE ALL instances with their descriptions
5. Do NOT filter out or consolidate observations - include ALL potentially positive findings

If absolutely no positive findings are in the list (zero mentions of visible markers), then state: "No specific positive iridology markers were clearly identified in the prior observations."

Be comprehensive and include EVERY possible marker mentioned. This is critical for accurate analysis.
"""

# REVISED SYNTHESIS PROMPT: Now expects a pre-summarized list of positive findings
SYNTHESIS_PROMPT_TEMPLATE = """
The following is a list of *key positive iridology signs* that were identified from an iris image:
--- BEGIN KEY POSITIVE SIGNS LIST ---
{summarized_positive_findings}
--- END KEY POSITIVE SIGNS LIST ---

CRITICAL INSTRUCTIONS: 
1. If the list above contains ANY specific iridology markers (like "Cholesterol ring", "Stress ring", etc.), you MUST analyze these specific markers in detail and build your entire analysis around them.
2. ONLY provide a generic analysis if the list explicitly states "No specific positive iridology markers were clearly identified".
3. If the list begins with "IMPORTANT POSITIVE FINDINGS", these are raw observations that MUST be treated as valid findings.

Based ONLY on the items in the 'KEY POSITIVE SIGNS LIST' above, provide a comprehensive analysis in the following format:

1.  **QUICK SUMMARY TABLE**
    | Category | Sign | Confidence Level | Key Implication |
    |----------|------|-----------------|-----------------|
    [Create a table with the most significant findings, confidence levels (High/Medium/Low), and brief implications]

2.  **DETAILED IRIS ANALYSIS**
    a.  **Key Positive Signs**
        * List all observed signs exactly as provided in the 'KEY POSITIVE SIGNS LIST'
        * If no positive signs were found, state: "No specific positive iridology markers were clearly identified in this analysis."

    b.  **Detailed Analysis by Sign**
        For each identified sign (if any):
        * Sign Name: [Name of the sign]
        * Visual Description: [Detailed description of what was observed]
        * Location: [Specific area in the iris]
        * Confidence Level: [High/Medium/Low] with brief explanation
        * Potential Implications: [Specific body system or area of focus]
        * Relevance: [Brief explanation of why this sign is significant]

3.  **COMPREHENSIVE ASSESSMENT**
    a.  **Key Findings Summary:**
        * Provide a concise summary of the most significant findings
        * Highlight any patterns or connections between different signs
        * Note any areas of particular concern or interest
    
    b.  **Wellness Recommendations:**
        * Primary Recommendations (2-3):
          - Specific, actionable suggestions
          - Directly related to the most significant findings
          - Include timing and frequency where applicable
        
        * Supporting Practices (2-3):
          - Complementary wellness activities
          - Lifestyle adjustments
          - Dietary considerations
    
    c.  **Follow-up Recommendations:**
        * Short-term (1-2 weeks):
          - Immediate actions to take
          - What to monitor
        
        * Medium-term (1-3 months):
          - Ongoing practices
          - Progress indicators
        
        * Long-term (3-6 months):
          - Sustained lifestyle changes
          - When to reassess

4.  **ADDITIONAL NOTES**
    * Any relevant observations about image quality or clarity
    * Areas that might benefit from additional analysis
    * General wellness context

Format the output with:
- Clear section headers
- Consistent bullet points
- Proper spacing
- Professional language
- Tables where appropriate
- Emphasis on actionable items

Maintain a professional, supportive tone throughout the analysis.
"""

MARKER_PROMPT_TEMPLATE = """
Analyze this iris image for the marker '{marker_name}'. 

As an iridology expert, examine:
1. Inner iris: Digestive system
2. Middle iris: Major organs
3. Outer iris: Skin, lymphatics

Focus only on '{marker_name}' and respond with:
KEY FINDING: (YES/NO/POSSIBLE)
CONFIDENCE: (High/Medium/Low)
VISUAL DETAILS:
  - LOCATION: Where in the iris
  - APPEARANCE: Colors and patterns
  - SIZE: How large/extensive
SELF-VERIFICATION: (YES/NO)

Be specific and concise.
"""

# --- 3. Interact with LM Studio LLaVA API ---
# REFACTORED get_llava_response to use llava_mlx_client
def get_llava_response(image_base64, prompt_text, model_name=MODEL_NAME, current_max_tokens=768):
    temp_image_path = None
    try:
        if image_base64 and image_base64 != "None":
            # Decode base64 and save to a temporary file
            try:
                image_bytes = base64.b64decode(image_base64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(image_bytes)
                    temp_image_path = tmp_file.name
            except Exception as e:
                error_msg = f"Error: Failed to decode/save temporary image from base64 - {e}"
                print(error_msg)
                return error_msg
        # If image_base64 is None or "None", temp_image_path remains None
        
        # Call the client's query_llava function
        # client_query_llava expects: image_path, prompt, api_url, model, max_tokens, temperature
        response_content = client_query_llava(
            image_path=temp_image_path, # This will be None if image_base64 was None or "None"
            prompt=prompt_text,
            api_url=LM_STUDIO_API_URL, 
            model=model_name, 
            max_tokens=current_max_tokens,
            temperature=0.1 # Default from old payload, client default is also 0.1
        )
        return response_content

    except Exception as e:
        error_msg = f"Error in new get_llava_response wrapper: {e}"
        print(error_msg)
        return error_msg
    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary image file {temp_image_path} - {e}")

def encode_image_to_base64(image_path_or_bytes):
    try:
        if isinstance(image_path_or_bytes, str):
            with open(image_path_or_bytes, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_path_or_bytes, bytes):
             return base64.b64encode(image_path_or_bytes).decode('utf-8')
        else:
            raise ValueError("Input must be a file path (string) or image bytes.")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path_or_bytes}")
        return None
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# --- 4. Main Orchestration Logic ---
def analyze_iris_image(image_path_or_bytes, iridology_checklist):
    if not iridology_checklist:
        return "Error: Iridology checklist is empty. Cannot perform analysis."

    image_base64 = encode_image_to_base64(image_path_or_bytes)
    if not image_base64:
        return "Error: Could not encode image."

    print(f"\n--- Starting Analysis for Image ---")
    all_raw_probe_observations = [] 
    significant_observations_count = 0  # Track number of potentially significant observations

    # Step 1: Probing
    print("\n--- Stage 1: Detailed Probing ---")
    for i, sign_info in enumerate(iridology_checklist):
        main_cat = sign_info.get('Main Category', 'N/A')
        sub_cat = sign_info.get('Sub-Category', 'N/A')
        markers = sign_info.get('Analysis(*)', 'No specific markers listed.') 

        main_cat_str = str(main_cat) if pd.notna(main_cat) else "N/A"
        sub_cat_str = str(sub_cat) if pd.notna(sub_cat) else "N/A"
        markers_str = str(markers) if pd.notna(markers) else "No specific markers listed."

        probing_prompt = PROBING_PROMPT_TEMPLATE.format(
            main_category=main_cat_str,
            sub_category=sub_cat_str,
            detailed_markers=markers_str
        )
        print(f"\n[Probe {i+1}/{len(iridology_checklist)}] Asking about: {sub_cat_str} (Guided by: {markers_str[:70]}...)")
        # Allow slightly more tokens for probes to capture detail
        observation = get_llava_response(image_base64, probing_prompt, current_max_tokens=200) 
        print(f"LLaVA's Raw Observation for '{sub_cat_str}': {observation[:250]}...")
        
        # Track if this observation might contain significant findings
        if "OBSERVED" in observation or "observed" in observation.lower() or "visible" in observation.lower():
            significant_observations_count += 1
            
        all_raw_probe_observations.append(f"Observation regarding '{sub_cat_str}' (Expected: '{markers_str[:70]}...'):\n{observation}")

    # Stage 1.5: Summarize Probe Observations (Text-only call to LLaVA)
    print("\n--- Stage 1.5: Summarizing Probe Observations ---")
    raw_observations_text = "\n\n---\n\n".join(all_raw_probe_observations)
    print(f"Length of all raw probe observations for summarization: {len(raw_observations_text)} characters")
    print(f"Number of potentially significant observations detected: {significant_observations_count}")

    probe_summary_prompt = PROBE_SUMMARY_PROMPT_TEMPLATE.format(all_raw_probe_observations=raw_observations_text)
    # This is a text-only call, so image_base64 is None. Max tokens can be moderate.
    summarized_positive_findings = get_llava_response(None, probe_summary_prompt, current_max_tokens=1024)
    
    print(f"\nLLaVA's Summarized Positive Findings:\n{summarized_positive_findings}")
    print(f"Length of summarized positive findings: {len(summarized_positive_findings)} characters")

    # VERIFICATION STEP: Check if summary is potentially missing observations
    if "Error:" in summarized_positive_findings:
         return f"Failed during probe summarization step: {summarized_positive_findings}"
         
    if ("No specific positive iridology markers" in summarized_positive_findings and 
        significant_observations_count > 0):
        print("\nWARNING: Summary indicates no markers, but raw observations showed potential findings.")
        print("Adding raw significant observations to ensure findings aren't lost...")
        
        # Extract and append the potentially significant observations directly
        significant_obs = []
        for obs in all_raw_probe_observations:
            if "OBSERVED" in obs or "observed" in obs.lower() or "visible" in obs.lower():
                # Extract just the actual observation, not the metadata
                obs_parts = obs.split(":\n", 1)
                if len(obs_parts) > 1:
                    significant_obs.append(obs_parts[1].strip())
                else:
                    significant_obs.append(obs)
        
        if significant_obs:
            summarized_positive_findings = "IMPORTANT POSITIVE FINDINGS:\n\n" + "\n\n".join(significant_obs)
            print(f"\nRevised Findings (including raw observations):\n{summarized_positive_findings[:500]}...")

    # Step 2: Final Synthesis using the LLaVA-summarized positive findings
    print("\n--- Stage 2: Final Synthesis ---")
    synthesis_prompt_text = SYNTHESIS_PROMPT_TEMPLATE.format(summarized_positive_findings=summarized_positive_findings)
    print(f"Total length of final synthesis prompt text (template + summarized findings): {len(synthesis_prompt_text)} characters")
    
    if len(synthesis_prompt_text) > 13000: 
        print("WARNING: Final synthesis prompt text is very long and might exceed token limits even after summarization.")

    print("\nAsking LLaVA to synthesize the final analysis...")
    # Image is included again for the final synthesis for context, LLaVA might re-verify.
    final_analysis = get_llava_response(image_base64, synthesis_prompt_text, current_max_tokens=1500) 

    print("\n--- Analysis Complete ---")
    return final_analysis

def analyze_iris_image_marker_by_marker(image_path_or_bytes, iridology_checklist):
    """
    Analyzes an iris image by checking for each marker individually.
    
    Args:
        image_path_or_bytes: Path to image file or bytes of the image
        iridology_checklist: List of dictionaries containing iridology markers to check
        
    Returns:
        List of dictionaries with marker responses
    """
    if not iridology_checklist:
        return "Error: Iridology checklist is empty. Cannot perform analysis."

    image_base64 = encode_image_to_base64(image_path_or_bytes)
    if not image_base64:
        return "Error: Could not encode image."

    # Create iridology reference file if it doesn't exist
    reference_file = 'iridology_reference.csv'
    if not os.path.exists(reference_file):
        reference_file = create_iridology_reference()
    
    # Load reference data to include in prompts - SIMPLIFIED
    reference_context = ""
    try:
        import csv
        with open(reference_file, 'r') as f:
            # Skip loading the full reference to keep prompt shorter
            reference_context = "Reference data available but omitted to conserve tokens."
    except Exception as e:
        print(f"Warning: Could not load reference data: {e}")
        reference_context = ""

    print(f"\n--- Starting Marker-by-Marker Analysis for Image ---")
    marker_responses = []

    # Map markers to body systems for better categorization
    body_systems = {
        "digestive": ["digestive", "stomach", "intestine", "bowel", "colon", "gallbladder", 
                     "liver", "pancreas", "acid", "digestion", "absorption", "reflux"],
        "nervous": ["nervous", "stress", "anxiety", "tension", "mental", "pineal", "melatonin", 
                   "sleep", "nerve", "brain", "headache", "emotional"],
        "endocrine": ["hormone", "thyroid", "adrenal", "pituitary", "insulin", "pancreas", 
                     "blood sugar", "metabolism", "energy"],
        "circulatory": ["circulatory", "heart", "blood", "circulation", "cardiovascular", 
                        "cholesterol", "artery", "vein", "oxygen", "iron"],
        "lymphatic": ["lymph", "lymphatic", "immune", "tonsil", "spleen", "toxin", "drainage",
                     "congestion", "inflammation"],
        "respiratory": ["respiratory", "lung", "breathing", "breath", "chest", "air", "respiration"],
        "structural": ["structural", "spine", "neck", "joint", "knee", "posture", "alignment", 
                      "muscle", "bone", "back"]
    }

    for i, sign_info in enumerate(iridology_checklist):
        marker_name = sign_info.get('Analysis(*)', 'No specific marker listed.')
        main_category = sign_info.get('Main Category', '')
        sub_category = sign_info.get('Sub-Category', '')
        
        if not marker_name or marker_name.strip() == "":
            continue
            
        # Include category information for better context
        full_marker_name = marker_name
        if main_category and sub_category:
            # Simplify categorized name to save tokens
            categorized_name = f"{marker_name} [{main_category}]"
            # Keep it shorter to save tokens
            if len(categorized_name) < 60:
                full_marker_name = categorized_name
        
        # Determine which body system this marker belongs to
        body_system = "other"
        max_matches = 0
        text_to_check = f"{marker_name} {main_category} {sub_category}".lower()
        for system, keywords in body_systems.items():
            matches = sum(1 for keyword in keywords if keyword.lower() in text_to_check)
            if matches > max_matches:
                max_matches = matches
                body_system = system
        
        # Format the prompt with the marker name - SIMPLIFIED, NO REFERENCE DATA
        enhanced_marker_prompt = MARKER_PROMPT_TEMPLATE.format(marker_name=full_marker_name)
        
        print(f"\n[Marker {i+1}/{len(iridology_checklist)}] Analyzing: {full_marker_name[:70]}...")
        # Reduced max tokens to avoid context window issues
        response = get_llava_response(image_base64, enhanced_marker_prompt, current_max_tokens=200)
        
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
        
        marker_responses.append({
            "marker": full_marker_name,
            "main_category": main_category,
            "sub_category": sub_category,
            "body_system": body_system,
            "response": response,
            "is_generic": is_generic
        })

    # Log debug info about categorization results
    system_counts = {}
    for system in body_systems.keys():
        system_counts[system] = sum(1 for r in marker_responses if r.get("body_system") == system)
    system_counts["other"] = sum(1 for r in marker_responses if r.get("body_system") == "other")
    
    print("\n--- Body System Categorization Results ---")
    for system, count in system_counts.items():
        print(f"  • {system.capitalize()}: {count} markers")
    print("----------------------------------------")
    
    return marker_responses

def aggregate_marker_responses(marker_responses):
    """
    Aggregates and formats marker responses into a concise report.
    
    Args:
        marker_responses (list): List of dictionary responses from marker analysis
        
    Returns:
        str: Formatted report with flagged items
    """
    abnormal_findings = []
    normal_findings = []
    uncertain_findings = []
    possible_findings = []  # New category for POSSIBLE findings
    
    for resp in marker_responses:
        marker_name = resp["marker"]
        response = resp["response"]
        
        # Extract the key finding and confidence using our structured format
        key_finding = "No clear finding"
        confidence = "Uncertain"
        is_abnormal = False
        is_uncertain = False
        is_possible = False
        is_generic = is_generic_response(response)
        
        # Try to extract structured data
        if "KEY FINDING:" in response:
            key_finding_section = response.split("KEY FINDING:")[1].split("\n")[0].strip()
            key_finding = key_finding_section
            
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
            
        # Check if this is an abnormal, normal, possible, or uncertain finding
        key_finding_lower = key_finding.lower()
        
        if "not observed" in key_finding_lower or "no " in key_finding_lower or "absent" in key_finding_lower:
            is_abnormal = False
        elif "possible" in key_finding_lower or "may be" in key_finding_lower:
            is_possible = True
            is_abnormal = False  # Not definitively abnormal
        elif "yes" in key_finding_lower or "present" in key_finding_lower or "observed" in key_finding_lower:
            is_abnormal = True
        else:
            is_uncertain = True
            
        # If self-verification is No, treat as uncertain regardless
        if self_verified.lower() == "no":
            is_uncertain = True
            
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
        }
        
        # Sort into appropriate category
        if is_generic or is_uncertain:
            uncertain_findings.append(entry)
        elif is_possible:
            possible_findings.append(entry)
        elif is_abnormal:
            abnormal_findings.append(entry)
        else:
            normal_findings.append(entry)
    
    # Create the report
    report = "IRIDOLOGY ANALYSIS SUMMARY\n"
    report += "==========================\n\n"
    
    # Add quick statistics
    report += f"STATISTICS:\n"
    report += f"• Definitively Abnormal: {len(abnormal_findings)} markers\n"
    report += f"• Possibly Abnormal: {len(possible_findings)} markers\n"
    report += f"• Normal/Absent: {len(normal_findings)} markers\n"
    report += f"• Uncertain/Flagged: {len(uncertain_findings)} markers\n\n"
    
    # Add abnormal findings summary if any exist
    if abnormal_findings:
        report += "KEY ABNORMAL FINDINGS (Definitive):\n"
        for i, finding in enumerate(abnormal_findings):
            report += f"⚠️ {i+1}. {finding['marker'].split('[')[0].strip()}: {finding['key_finding']} (Confidence: {finding['confidence']})\n"
        report += "\n"
    else:
        report += "No definitive abnormal findings detected.\n\n"
    
    # Add possible findings section
    if possible_findings:
        report += "POSSIBLE ABNORMAL FINDINGS:\n"
        for i, finding in enumerate(possible_findings):
            report += f"? {i+1}. {finding['marker'].split('[')[0].strip()}: {finding['key_finding']} (Confidence: {finding['confidence']})\n"
        report += "\n"
        
    # Add uncertain findings that were flagged
    if uncertain_findings:
        report += "UNCERTAIN/FLAGGED RESPONSES:\n"
        for i, finding in enumerate(uncertain_findings):
            marker = finding['marker'].split('[')[0].strip()
            report += f"⚠️ {i+1}. {marker}: FLAGGED - {finding['key_finding']}\n"
            # Include partial finding if available
            if finding['key_finding'] != "No clear finding":
                report += f"   Partial finding: {finding['key_finding']}\n"
        report += "\n"
    
    # Add detailed section header
    report += "DETAILED ANALYSIS:\n"
    report += "=================\n\n"
    
    # Add abnormal findings section
    if abnormal_findings:
        report += "DEFINITIVE ABNORMAL MARKERS:\n"
        for finding in abnormal_findings:
            marker_simple_name = finding['marker'].split('[')[0].strip()
            category_info = ""
            if '[' in finding['marker'] and ']' in finding['marker']:
                category_info = finding['marker'].split('[')[1].split(']')[0].strip()
            
            report += f"• {marker_simple_name}:\n"
            if category_info:
                report += f"  Category: {category_info}\n"
            report += f"  Finding: {finding['key_finding']}\n"
            report += f"  Confidence: {finding['confidence']}\n"
            report += f"  Details: {finding['visual_details']}\n\n"
    
    # Add possible findings section with details
    if possible_findings:
        report += "POSSIBLE ABNORMAL MARKERS:\n"
        for finding in possible_findings:
            marker_simple_name = finding['marker'].split('[')[0].strip()
            category_info = ""
            if '[' in finding['marker'] and ']' in finding['marker']:
                category_info = finding['marker'].split('[')[1].split(']')[0].strip()
            
            report += f"• {marker_simple_name}:\n"
            if category_info:
                report += f"  Category: {category_info}\n"
            report += f"  Finding: {finding['key_finding']}\n"
            report += f"  Confidence: {finding['confidence']}\n"
            report += f"  Details: {finding['visual_details']}\n\n"
        
    # Add normal findings section
    if normal_findings:
        report += "NORMAL/NEGATIVE MARKERS:\n"
        for finding in normal_findings:
            marker_simple_name = finding['marker'].split('[')[0].strip()
            report += f"• {marker_simple_name}: {finding['key_finding']} (Confidence: {finding['confidence']})\n"
        report += "\n"
    
    # Add a brief wellness recommendation section
    report += "RECOMMENDATIONS:\n"
    all_potential_issues = abnormal_findings + possible_findings
    if abnormal_findings:
        report += "Based on the definitively abnormal markers identified, consider consulting with a health professional\n"
        report += "about the specific findings, particularly: " + ", ".join([f['marker'].split('[')[0].strip() for f in abnormal_findings[:3]]) + ".\n\n"
    elif possible_findings and not abnormal_findings:
        report += "Several markers were identified as possibly abnormal. Consider monitoring these areas and potentially\n"
        report += "follow up with additional analysis: " + ", ".join([f['marker'].split('[')[0].strip() for f in possible_findings[:3]]) + ".\n\n"
    elif uncertain_findings and not all_potential_issues:
        report += "Several markers were flagged as uncertain. Consider a follow-up analysis with improved imaging.\n\n"
    else:
        report += "No specific markers of concern were identified in this analysis.\n\n"
    
    report += "Note: This analysis is based on iridology principles and is not a medical diagnosis.\n"
    
    return report

def is_generic_response(response):
    """Check if the response is generic or uncertain."""
    if not response or "Error:" in response or "API request failed" in response:
        return True
    
    try:
        # Attempt to extract the finding and confidence sections
        if "KEY FINDING:" in response:
            finding_line = response.split("KEY FINDING:")[1].split("\n")[0].strip().lower()
        else:
            return True  # No key finding section
            
        if "CONFIDENCE:" in response:
            confidence_line = response.split("CONFIDENCE:")[1].split("\n")[0].strip().lower()
        else:
            return True  # No confidence section
        
        if "VISUAL DETAILS:" in response:
            visual_details = response.split("VISUAL DETAILS:")[1].split("\n\n")[0].strip().lower()
        else:
            return True  # No visual details section
            
        # Check for generic or uncertain responses
        if "unsure" in finding_line or "unclear" in finding_line or "cannot" in finding_line or "possible" in finding_line:
            return True
            
        if "low" in confidence_line:
            return True
            
        # Check for generic visual descriptions
        generic_terms = [
            "the eye has a yellowish tint", 
            "yellowish-brown iris", 
            "the eye appears to be",
            "the iris is", 
            "the pupil is",
            "dark spot in the center",
            "black center",
            "black pupil",
            "surrounded by",
            "unusual appearance",
            "unusual coloration",
            "unusual feature"
        ]
        
        # If the visual details only contain generic terms
        if any(term in visual_details for term in generic_terms) and all(
            marker_term not in visual_details for marker_term in [
                "ring", "lacuna", "crypt", "pigment", "spot", "fiber", 
                "radial", "furrow", "nerve", "wreath", "zone", "lymphatic",
                "autonomic", "ciliary", "pupillary", "limbus"
            ]):
            return True
            
        # Check if the visual details are too short (generic descriptions tend to be short)
        if len(visual_details) < 50:  # Arbitrary threshold, adjust as needed
            return True
            
        # Look for required structuring in response (should mention location and appearance)
        if "location:" not in visual_details.lower() and "appearance:" not in visual_details.lower():
            return True
            
        return False
    except Exception as e:
        print(f"Error parsing response: {e}")
        return True  # If there's any error parsing, assume it's generic

def batch_process_images(image_dir='images', output_dir='analysis_results'):
    """
    Process all images in the specified directory using the marker-by-marker approach.
    
    Args:
        image_dir (str): Directory containing the iris images
        output_dir (str): Directory to save the analysis results
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load the iridology checklist once for all images
    checklist = load_iridology_data(IRIDOLOGY_DATA_FILE)
    if not checklist:
        print("Could not load the iridology checklist. Aborting batch analysis.")
        return
    
    # Get all jpg files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
    
    print(f"Found {len(image_files)} images to process in {image_dir}")
    
    # Verify if the LLaVA model is active before proceeding
    print("Verifying LLaVA model before batch processing...")
    if not verify_active_model():
        print("⚠️ WARNING: Could not confirm LLaVA model is active. Batch processing may fail.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Batch processing aborted.")
            return
    
    # Process each image
    batch_start_time = time.time()
    summary_report = f"IRIDOLOGY BATCH ANALYSIS SUMMARY\n"
    summary_report += f"Using LLaVA Model: {MODEL_NAME}\n"
    summary_report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary_report += f"All images resized to {DEFAULT_RESIZE_WIDTH}x{DEFAULT_RESIZE_HEIGHT} to reduce token count\n"
    summary_report += f"=================================\n\n"
    
    # Statistics tracking
    total_markers_analyzed = 0
    total_abnormal_findings = 0
    total_possible_findings = 0
    total_normal_findings = 0
    total_uncertain_findings = 0
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        # Calculate estimated time
        if i > 0:
            elapsed_time = time.time() - batch_start_time
            avg_time_per_image = elapsed_time / i
            remaining_images = len(image_files) - i
            est_remaining_time = avg_time_per_image * remaining_images
            est_completion_time = datetime.now() + datetime.timedelta(seconds=est_remaining_time)
            
            print(f"\n\n[{i+1}/{len(image_files)}] Processing image: {image_file}")
            print(f"Elapsed time: {elapsed_time:.1f} sec | Est. completion: {est_completion_time.strftime('%H:%M:%S')}")
        else:
            print(f"\n\n[{i+1}/{len(image_files)}] Processing image: {image_file}")
        
        # Skip if file not found or not readable
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            summary_report += f"Image {image_file}: ERROR - File not found\n\n"
            continue
        
        try:
            # ADDED: Resize the image before processing to reduce token count
            original_image_path = image_path
            resized_image_path = resize_image_before_processing(image_path)
            
            # Process the image
            image_start_time = time.time()
            marker_responses = analyze_iris_image_marker_by_marker(resized_image_path, checklist)
            
            # Generate the report
            report = aggregate_marker_responses(marker_responses)
            image_processing_time = time.time() - image_start_time
            
            # Save the report to a file
            report_file = os.path.join(output_dir, f"{image_file.split('.')[0]}_analysis.txt")
            with open(report_file, 'w') as f:
                f.write(f"Analysis of image: {image_file} (resized to {DEFAULT_RESIZE_WIDTH}x{DEFAULT_RESIZE_HEIGHT})\n")
                f.write(f"Processing time: {image_processing_time:.2f} seconds\n")
                f.write(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report)
            
            # Extract key statistics
            abnormal_count = sum(1 for r in marker_responses if 
                "yes" in r["response"].lower() and not is_generic_response(r["response"]))
            
            possible_count = sum(1 for r in marker_responses if 
                "possible" in r["response"].lower() and not is_generic_response(r["response"]))
                
            normal_count = sum(1 for r in marker_responses if 
                ("no" in r["response"].lower() or "not observed" in r["response"].lower()) 
                and not is_generic_response(r["response"]))
                
            uncertain_count = sum(1 for r in marker_responses if is_generic_response(r["response"]))
            
            # Update overall statistics
            total_markers_analyzed += len(marker_responses)
            total_abnormal_findings += abnormal_count
            total_possible_findings += possible_count
            total_normal_findings += normal_count
            total_uncertain_findings += uncertain_count
            
            # Add to the summary report
            summary_report += f"Image {image_file}:\n"
            summary_report += f"• Processing time: {image_processing_time:.1f} seconds\n"
            summary_report += f"• Definitively abnormal markers: {abnormal_count}\n"
            summary_report += f"• Possibly abnormal markers: {possible_count}\n"
            summary_report += f"• Normal/negative markers: {normal_count}\n"
            summary_report += f"• Uncertain/generic responses: {uncertain_count}\n"
            
            # Add key abnormal findings to summary
            if abnormal_count > 0 or possible_count > 0:
                summary_report += "• Key findings:\n"
                
                # First list definitive findings
                for r in marker_responses:
                    if "yes" in r["response"].lower() and not is_generic_response(r["response"]):
                        marker_name = r["marker"].split('[')[0].strip()
                        key_finding = "Detected"
                        if "KEY FINDING:" in r["response"]:
                            key_finding = r["response"].split("KEY FINDING:")[1].split("\n")[0].strip()
                        summary_report += f"  - DEFINITIVE: {marker_name}: {key_finding}\n"
                
                # Then list possible findings
                for r in marker_responses:
                    if "possible" in r["response"].lower() and not is_generic_response(r["response"]):
                        marker_name = r["marker"].split('[')[0].strip()
                        key_finding = "Possibly detected"
                        if "KEY FINDING:" in r["response"]:
                            key_finding = r["response"].split("KEY FINDING:")[1].split("\n")[0].strip()
                        summary_report += f"  - POSSIBLE: {marker_name}: {key_finding}\n"
            
            summary_report += "\n"
            
            print(f"Completed analysis for {image_file} in {image_processing_time:.1f} seconds")
            print(f"Report saved to {report_file}")
            print(f"Found: {abnormal_count} definitive, {possible_count} possible, {normal_count} normal, {uncertain_count} uncertain")
            
            # Clean up the temporary resized image if it's different from the original
            if resized_image_path != original_image_path and os.path.exists(resized_image_path):
                try:
                    os.remove(resized_image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {resized_image_path}: {e}")
            
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            summary_report += f"Image {image_file}: ERROR - {str(e)}\n\n"
    
    # Add overall statistics to summary report
    total_time = time.time() - batch_start_time
    avg_time = total_time / len(image_files) if image_files else 0
    
    summary_report += "\nBATCH PROCESSING STATISTICS:\n"
    summary_report += f"• Total images processed: {len(image_files)}\n"
    summary_report += f"• Total processing time: {total_time:.1f} seconds\n"
    summary_report += f"• Average time per image: {avg_time:.1f} seconds\n"
    summary_report += f"• Total markers analyzed: {total_markers_analyzed}\n"
    summary_report += f"• Total definitive abnormal findings: {total_abnormal_findings}\n"
    summary_report += f"• Total possible abnormal findings: {total_possible_findings}\n"
    summary_report += f"• Total normal findings: {total_normal_findings}\n"
    summary_report += f"• Total uncertain/generic responses: {total_uncertain_findings}\n"
    
    if total_markers_analyzed > 0:
        uncertain_percentage = (total_uncertain_findings / total_markers_analyzed) * 100
        summary_report += f"• Uncertain response rate: {uncertain_percentage:.1f}%\n"
    
    # Save the summary report
    summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write(summary_report)
    
    print(f"\nBatch processing complete!")
    print(f"Processed {len(image_files)} images in {total_time:.2f} seconds (avg: {avg_time:.2f} sec/image)")
    print(f"Summary report saved to {summary_file}")
    
    return summary_file


def verify_active_model():
    """Check if the active LLaVA model matches our configuration."""
    # Just a simple text query with no image to check if the API is working
    try:
        # Using None as the image_base64 to avoid sending an image - pure text query
        response = get_llava_response(None, "What LLaVA model version are you?", current_max_tokens=50)
        print("Active LLaVA Model Check:")
        print(f"Response to model check: {response}")
        
        # Check if the response contains the expected model info
        # Loosened check to just "llava" as model might report full internal name
        if "llava" in response.lower(): 
            print(f"✅ Confirmed LLaVA model is active (response contains 'llava')")
            # Further check if MODEL_NAME (e.g., "llava-v1.6-vicuna-13b") is in response for more specific check
            if MODEL_NAME.lower() in response.lower():
                 print(f"✅ Specifically, model response indicates: {MODEL_NAME}")
            else:
                print(f"ℹ️ Note: Specific model name '{MODEL_NAME}' not in response, but 'llava' was found.")
            return True
        else:
            print(f"⚠️ Could not confirm LLaVA model is active based on response: {response}")
            return False
    except Exception as e:
        print(f"❌ Error checking LLaVA model: {e}")
        return False

def create_iridology_reference():
    """
    Creates a CSV file with iridology reference mapping zones to body systems.
    This will be used to enhance the model's understanding of iridology.
    """
    reference_data = [
        ["Zone", "Position", "Body System", "Common Markers", "Appearance"],
        ["Pupillary Zone", "Inner 1/3 of iris", "Digestive System", "Stomach ring, intestinal markers", "Often lighter in color, may show digestive weakness through white hue"],
        ["Ciliary Zone", "Middle 1/3 of iris", "Major organs, Circulation", "Organ lesions, pigmentation", "Contains most organ representations, shows structural signs"],
        ["Autonomic Nerve Wreath", "Between Pupillary and Ciliary", "Nervous System", "Stress rings, nerve reflexes", "Appears as a jagged or circular boundary, indicates ANS health"],
        ["Limbus/Iris Edge", "Outer edge of iris", "Skin, Lymphatic System", "Lymphatic rosary, scurf rim", "Outer edge; darkness or white ring indicates issues"],
        ["Collarette", "Ridge around pupillary zone", "Digestive System Nerve Supply", "Autonomic nerve wreath", "Circular ridge, should be regular and well-defined"],
        ["1 o'clock", "Top right (right eye)", "Brain, Head", "Brain lesions, headache signs", "Marks in this area relate to head and brain"],
        ["2 o'clock", "Right eye", "Pituitary, Pineal, Right side of head", "Hormone markers, pituitary signs", "Relates to endocrine master glands"],
        ["3 o'clock", "Right side (right eye)", "Right ear, neck, shoulder, arm", "Neck tension signs, ear markers", "Right side peripheral issues"],
        ["4-5 o'clock", "Lower right (right eye)", "Liver, Gallbladder", "Liver spots, gallbladder cloudiness", "Often shows as yellow/brown discoloration"],
        ["6 o'clock", "Bottom (right eye)", "Small Intestine, Ascending Colon", "Bowel pockets, crypts", "Lower segment relating to elimination"],
        ["7-8 o'clock", "Lower left (right eye)", "Genitourinary System", "Reproductive organ marks", "Relates to reproductive and urinary systems"],
        ["9 o'clock", "Left side (right eye)", "Left side chest, heart", "Heart stress signs", "Left mid-outer area in right eye"],
        ["10-11 o'clock", "Upper left (right eye)", "Thyroid, Left shoulder", "Thyroid arc, shoulder tension", "Upper left quadrant structural markers"],
        ["12 o'clock", "Top (both eyes)", "Brain, Cerebral circulation", "Circulation markers, brain signs", "Top segment in both eyes"],
        ["Stress Rings", "Concentric white rings", "Nervous System Stress", "Anxiety, tension", "White concentric circles radiating from pupil"],
        ["Lacunae", "Closed dark spots/pits", "Inherent Weakness", "Organ lesions", "Dark, enclosed spaces indicating tissue damage"],
        ["Radii Solaris", "Straight lines radiating outward", "Toxic Stress Lines", "Toxicity, inflammation", "Straight spoke-like lines extending from pupil"],
        ["Pigmentation", "Colored deposits", "Toxicity, Medication", "Drug deposits, metal toxicity", "Brown, orange, yellow spots indicating specific substances"],
        ["Crypts", "Open, white lesions", "Acute Activity", "Inflammation, infection", "White openings indicating current active processes"]
    ]
    
    # Write the reference data to a CSV file
    import csv
    with open('iridology_reference.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reference_data)
    
    print("Created iridology reference file: iridology_reference.csv")
    return 'iridology_reference.csv'

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

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        # Batch process all images
        print("Starting batch processing of all images...")
        batch_process_images()
    else:
        # Process a single test image
        # MODIFIED to specifically use images/16.jpg for testing
        test_image_path = 'images/16.jpg'
        
        if not os.path.exists(test_image_path):
            print(f"Error: Test image '{test_image_path}' not found.")
            # Fallback if 16.jpg is somehow missing, though it shouldn't be
            print("Falling back to first image in 'images' directory if available.")
            image_files = [f for f in os.listdir('images') if f.lower().endswith('.jpg')]
            if image_files:
                image_files.sort()
                test_image_path = os.path.join('images', image_files[0])
                print(f"Using fallback image: {test_image_path}")
            else:
                print("No images found in 'images' directory. Please provide a valid image.")
                sys.exit(1)
        else:
            print(f"Using test image: {test_image_path}")
            
        # ADDED: Resize the image before analysis to reduce token count
        original_image_path = test_image_path
        test_image_path = resize_image_before_processing(test_image_path)

        # Check which model is active
        verify_active_model()

        # Determine if we should use hierarchical or flat approach based on command line args
        use_hierarchical = '--hierarchical' in sys.argv

        if use_hierarchical:
            print("Using hierarchical analysis approach")
            # Load data in hierarchical structure by body system
            hierarchical_data = load_iridology_data(
                IRIDOLOGY_DATA_FILE,
                priority_threshold=2,  # Include priority 1-2 (high and medium)
                max_markers=5,        # Up to 5 markers per body system
                hierarchical=True      # Return hierarchical structure
            )
            
            # Analyze using hierarchical approach
            print("\n--- Starting Hierarchical Analysis ---")
            marker_responses = analyze_iris_image_hierarchical(
                test_image_path, 
                hierarchical_data,
                max_markers_per_system=3,  # Analyze up to 3 markers per system
                early_stop_threshold=2     # Stop after finding 2 positive markers in a system
            )
        else:
            print("Using traditional marker-by-marker analysis approach")
            # Load data as flat list with priority filtering
            checklist = load_iridology_data(
                IRIDOLOGY_DATA_FILE, 
                priority_threshold=1,  # Only use high priority (1) markers
                max_markers=10         # Limit to 10 markers total
            )
            
            print(f"Using {len(checklist)} high-priority markers for analysis")
            
            # Use traditional marker-by-marker approach
            print("\n--- Starting Marker-by-Marker Analysis ---")
            marker_responses = analyze_iris_image_marker_by_marker(test_image_path, checklist)

        # Create final report from marker responses (works with both approaches)
        print("\n--- Creating Final Report ---")
        final_report = aggregate_marker_responses(marker_responses)

        print("\n\n--- FINAL IRIDOLOGY ANALYSIS ---")
        print(final_report)

        # Append results to results/iris_16_analysis.txt
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True) # Ensure results directory exists
        output_file_path = os.path.join(output_dir, "iris_16_analysis.txt")

        analysis_type = "Hierarchical" if use_hierarchical else "Traditional"
        with open(output_file_path, "a") as f: # Append mode
            f.write(f"\n\n--- {analysis_type} Analysis from iris_analyzer_backend.py ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Image: {original_image_path} (resized to {DEFAULT_RESIZE_WIDTH}x{DEFAULT_RESIZE_HEIGHT})\n\n")
            f.write(final_report)
        print(f"\nResults appended to: {output_file_path}")
        
        # Clean up the temporary resized image if it's different from the original
        if test_image_path != original_image_path and os.path.exists(test_image_path):
            try:
                os.remove(test_image_path)
                print(f"Cleaned up temporary resized image: {test_image_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {test_image_path}: {e}")
