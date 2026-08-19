"""
Community Plant Identification Manager
=======================================
Handles storage and retrieval of:
- Unknown plant submissions (low-confidence predictions)
- User-submitted plant names for unknowns
- Community identification requests from other users
"""

import json
import os
import uuid
from datetime import datetime

SUBMISSIONS_PATH = "community_submissions.json"


def _load_submissions():
    """Load community submissions from JSON file."""
    if os.path.exists(SUBMISSIONS_PATH):
        try:
            with open(SUBMISSIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"unknown_plants": [], "identification_requests": []}
    return {"unknown_plants": [], "identification_requests": []}


def _save_submissions(data):
    """Save community submissions to JSON file."""
    try:
        with open(SUBMISSIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError:
        return False


def add_unknown_plant(image_filename, model_prediction, model_confidence,
                      user_name="", user_provided_name="", context_data=None):
    """
    Add an unknown/low-confidence plant to the community gallery.
    """
    data = _load_submissions()
    submission_id = str(uuid.uuid4())[:8]

    entry = {
        "id": submission_id,
        "image_filename": image_filename,
        "model_prediction": model_prediction,
        "model_confidence": round(model_confidence, 2),
        "user_provided_name": user_provided_name,
        "submitted_by": user_name if user_name else "Anonymous",
        "status": "identified" if user_provided_name else "unidentified",
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "identification_suggestions": [],
        "context_data": context_data or {}
    }

    data["unknown_plants"].append(entry)
    _save_submissions(data)
    return submission_id


def submit_identification(plant_id, suggested_name, suggested_by=""):
    """
    Submit an identification suggestion for an unknown plant.

    Parameters
    ----------
    plant_id : str
        The submission ID of the unknown plant.
    suggested_name : str
        The suggested plant name.
    suggested_by : str
        Name of the person suggesting the identification.

    Returns
    -------
    bool
        True if the suggestion was recorded successfully.
    """
    data = _load_submissions()

    for plant in data["unknown_plants"]:
        if plant["id"] == plant_id:
            suggestion = {
                "suggested_name": suggested_name,
                "suggested_by": suggested_by if suggested_by else "Anonymous",
                "suggested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            plant["identification_suggestions"].append(suggestion)
            _save_submissions(data)
            return True

    return False


def get_unidentified_plants():
    """Return all plants that are still unidentified."""
    data = _load_submissions()
    return [p for p in data["unknown_plants"] if p["status"] == "unidentified"]


def get_all_submissions():
    """Return all community submissions."""
    data = _load_submissions()
    return data["unknown_plants"]


def get_submission_by_id(plant_id):
    """Return a specific submission by its ID."""
    data = _load_submissions()
    for plant in data["unknown_plants"]:
        if plant["id"] == plant_id:
            return plant
    return None
