import tkinter as tk
import pandas as pd
from tkinter import messagebox

def get_user_input_gui(csv_path, feature_name_map=None):
    # Get feature names
    df = pd.read_csv(csv_path)
    feature_names = [col for col in df.columns if col != 'PHQ9_Target']
    medians = df[feature_names].median()

    # Store multiple entries
    all_entries = []
    all_dates = []  
    final_result = None  

    def submit_entry():
        try:
            user_data = {}
            date_value = date_entry.get()

            # Check if the date field is empty
            if not date_value:
                raise ValueError("Date field cannot be empty. Please provide a valid date.")

            # Convert the date input to a datetime format
            user_data["date"] = pd.to_datetime(date_value, errors='coerce')
            if pd.isna(user_data["date"]):
                raise ValueError(f"Invalid date format: {date_value}. Please use YYYY-MM-DD.")
            
            # Collect feature data
            for feature, entry in entry_widgets.items():
                value = entry.get()
                if feature == "date":
                    continue  
                try:
                    user_data[feature] = float(value)  # Ensure numeric data
                except ValueError:
                    raise ValueError(f"Invalid numeric value: {value} for feature {feature}. Please enter a valid number.")

            all_entries.append(user_data)
            all_dates.append(user_data["date"])  # Store the date separately
            
            # Check contents 
            print("All Entries after adding this entry:", all_entries)
            
            messagebox.showinfo("Success", "Entry added successfully!")
            # Clear the entry fields
            for entry in entry_widgets.values():
                entry.delete(0, tk.END)
            date_entry.delete(0, tk.END)

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

    def finish():
        nonlocal final_result 
        if not all_entries:
            messagebox.showwarning("No data", "No entries were added.")
            return
        
        print("Final entries to process:", all_entries)
        
        final_df = pd.DataFrame(all_entries)
        date_series = pd.Series(all_dates).reset_index(drop=True)
        
        print("DataFrame after all entries:", final_df)
        print("Date Series:", date_series)
        
        final_result = (final_df, date_series, feature_names)  
        window.quit()  

    # Create GUI window
    window = tk.Tk()
    window.title("Enter User Data")

    # Date input
    tk.Label(window, text="Date (YYYY-MM-DD):").grid(row=0, column=0)
    date_entry = tk.Entry(window)
    date_entry.grid(row=0, column=1)

    # Input fields + median 
    entry_widgets = {}
    for idx, feature in enumerate(feature_names):
        row_num = idx + 1
        friendly_name = feature_name_map.get(feature, feature) if feature_name_map else feature
        tk.Label(window, text=friendly_name).grid(row=row_num, column=0)

        entry = tk.Entry(window)
        entry.grid(row=row_num, column=1)
        entry_widgets[feature] = entry

        # Show median value next to the entry
        median_val = round(medians[feature], 2)
        tk.Label(window, text=f"Median: {median_val}", fg="gray").grid(row=row_num, column=2)

    # Buttons
    tk.Button(window, text="Add Entry", command=submit_entry).grid(row=len(feature_names) + 1, column=0)
    tk.Button(window, text="Done", command=finish).grid(row=len(feature_names) + 1, column=1)

    window.mainloop()

    return final_result 
