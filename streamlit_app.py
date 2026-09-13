"""Streamlit Cloud entrypoint alias pointing to app.py"""
import runpy

if __name__ == "__main__":
    runpy.run_path("app.py", run_name="__main__")
