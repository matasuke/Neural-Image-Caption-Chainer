import requests
import json

def post_slack(url, name, text):
    requests.post(
        url,
        data=json.dumps(
            {
                "text": text,
                "username": name,
                "icon_emoji": "python:"
            }
        )
    )
