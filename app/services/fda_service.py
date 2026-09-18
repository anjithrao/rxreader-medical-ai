import requests


def get_drug_info(medicine_name):
    try:
        url = (
            "https://api.fda.gov/drug/label.json"
            f"?search=openfda.brand_name:{medicine_name}&limit=1"
        )
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "found": True,
                    "brand_name": result.get("openfda", {}).get("brand_name", ["N/A"])[0],
                    "generic_name": result.get("openfda", {}).get("generic_name", ["N/A"])[0],
                    "purpose": (
                        result.get("purpose", ["N/A"])[0][:200]
                        if result.get("purpose")
                        else "N/A"
                    ),
                    "warnings": (
                        result.get("warnings", ["N/A"])[0][:200]
                        if result.get("warnings")
                        else "N/A"
                    ),
                }
    except Exception:
        pass
    return {"found": False}
