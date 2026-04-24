# 🖥️ How to Run VayuDrishti on Your Own Computer

## 1. Prerequisites
- Python 3.8 or higher
- Git
- Internet connection (for installing dependencies)

## 2. Clone the Repository
Open PowerShell (Windows) or Terminal (Mac/Linux) and run:
```powershell
git clone https://github.com/nishant-gupta911/VayuDrishtii.git
cd VayuDrishtii
```

## 3. Create a Virtual Environment
This keeps dependencies isolated.
```powershell
python -m venv venv
```

## 4. Activate the Virtual Environment
- **Windows:**
  ```powershell
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

## 5. Install Dependencies
```powershell
pip install -r requirements.txt
```

## 6. Run the Dashboard
Navigate to the dashboard folder and start the app:
```powershell
cd dashboard
streamlit run dashboard.py
```

## 7. Access the Dashboard
Open your browser and go to:
```
http://localhost:8501
```

## 8. Troubleshooting
- If you see missing package errors, re-run `pip install -r requirements.txt`.
- Do **not** use the `venv` folder from GitHub—always create your own.
- For any issues, check the README or open an issue on GitHub.

---

**Enjoy using VayuDrishti!**
