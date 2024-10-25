import unittest
import pandas as pd
from scr.data_preprocessing import load_data, clean_data

class TestDataPreprocessing(unittest.TestCase):
    
    def test_load_data(self):
        data = load_data()
        self.assertIsInstance(data, pd.DataFrame)
    
    def test_clean_data(self):
        data = pd.DataFrame({
            'Age': [20, 30, None],
            'Income': [50000, None, 70000]
        })
        cleaned_data = clean_data(data)
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)  # Ensure no missing values
        
if __name__ == '__main__':
    unittest.main()
