"""
PRODIGY INFOTECH - MACHINE LEARNING INTERNSHIP
TASK-01: Interactive House Price Prediction CLI
"""

import os
import joblib
import pandas as pd

def predict_house_price(sqft, bedrooms, full_baths, half_baths=0):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "linear_regression_model.joblib")
    
    if not os.path.exists(model_path):
        print("Model file not found. Training model first...")
        from train_model import run_pipeline
        run_pipeline()

    model = joblib.load(model_path)
    
    input_data = pd.DataFrame([{
        "GrLivArea": float(sqft),
        "BedroomAbvGr": int(bedrooms),
        "FullBath": int(full_baths),
        "HalfBath": int(half_baths)
    }])
    
    predicted_price = model.predict(input_data)[0]
    return predicted_price

def interactive_cli():
    print("\n" + "=" * 60)
    print("      🏠 PRODIGY INFOTECH - HOUSE PRICE ESTIMATOR 🏠")
    print("=" * 60)
    print("Predict house price based on SqFt, Bedrooms & Bathrooms\n")
    
    try:
        sqft = float(input("Enter Living Area Square Footage (e.g. 1850): ").strip())
        bedrooms = int(input("Enter Number of Bedrooms (e.g. 3): ").strip())
        full_baths = int(input("Enter Number of Full Bathrooms (e.g. 2): ").strip())
        half_baths_input = input("Enter Number of Half Bathrooms [Default: 0]: ").strip()
        half_baths = int(half_baths_input) if half_baths_input else 0
        
        price = predict_house_price(sqft, bedrooms, full_baths, half_baths)
        
        print("\n" + "-" * 60)
        print(f"  📊 PROPERTY SPECIFICATIONS:")
        print(f"     - Living Area : {sqft:,.0f} sq ft")
        print(f"     - Bedrooms    : {bedrooms}")
        print(f"     - Full Baths  : {full_baths}")
        print(f"     - Half Baths  : {half_baths}")
        print("-" * 60)
        print(f"  💰 ESTIMATED HOUSE SALE PRICE: ${price:,.2f}")
        print("=" * 60 + "\n")
        
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter numeric values.")
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    interactive_cli()
