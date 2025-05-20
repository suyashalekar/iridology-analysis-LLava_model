#!/usr/bin/env python3
"""
Test script to verify configuration loading
"""

import os
import sys
import json
import unittest

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ConfigTest(unittest.TestCase):
    """Test case for configuration functionality"""
    
    def test_config_exists(self):
        """Test that the config file exists"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
        self.assertTrue(os.path.exists(config_path), "Config file does not exist")
    
    def test_config_loading(self):
        """Test that the config file can be loaded and contains required keys"""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for required keys
        self.assertIn('version', config, "Config missing 'version' key")
        self.assertIn('model', config, "Config missing 'model' key")
        self.assertIn('analysis', config, "Config missing 'analysis' key")
        self.assertIn('image_processing', config, "Config missing 'image_processing' key")
        self.assertIn('data', config, "Config missing 'data' key")
        
        # Check for specific model settings
        self.assertIn('name', config['model'], "Config missing 'model.name' key")
        self.assertIn('api_url', config['model'], "Config missing 'model.api_url' key")
        
        # Check for analysis settings
        self.assertIn('max_markers_per_system', config['analysis'], "Config missing 'analysis.max_markers_per_system' key")
        self.assertIn('early_stop_threshold', config['analysis'], "Config missing 'analysis.early_stop_threshold' key")
        
        # Check data file settings
        self.assertIn('iridology_data_file', config['data'], "Config missing 'data.iridology_data_file' key")
        
        # Verify the version format (should be semantic versioning)
        version = config['version']
        self.assertRegex(version, r'^\d+\.\d+\.\d+$', "Version does not follow semantic versioning (x.y.z)")

if __name__ == '__main__':
    unittest.main() 