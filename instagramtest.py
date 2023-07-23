import json
from typing import Dict
from urllib.parse import quote

import httpx

INSTAGRAM_APP_ID = "936619743392459"  # this is the public app id for instagram.com


def scrape_post(url_or_shortcode: str) -> Dict:
    """Scrape single Instagram post data"""
    if "http" in url_or_shortcode:
        shortcode = url_or_shortcode.split("/p/")[-1].split("/")[0]
    else:
        shortcode = url_or_shortcode
    print(f"scraping instagram post: {shortcode}")

    variables = {
        "shortcode": shortcode,
        "child_comment_count": 20,
        "fetch_comment_count": 100,
        "parent_comment_count": 24,
        "has_threaded_comments": True,
    }
    url = "https://www.instagram.com/graphql/query/?query_hash=b3055c01b4b222b8a47dc12b090e4e64&variables="
    result = httpx.get(
        url=url + quote(json.dumps(variables)),
        headers={"x-ig-app-id": INSTAGRAM_APP_ID},
    )
    data = json.loads(result.content)
    return data["data"]["shortcode_media"]

# Example usage:
posts = scrape_post("https://www.instagram.com/p/CuE2WNQs6vH/")
print(json.dumps(posts, indent=2, ensure_ascii=False))



# import json
# import httpx
# from urllib.parse import quote

# def scrape_user_posts(user_id: str, session: httpx.Client, page_size=12):
#     base_url = "https://www.instagram.com/graphql/query/?query_hash=e769aa130647d2354c40ea6a439bfc08&variables="
#     variables = {
#         "id": user_id,
#         "first": page_size,
#         "after": None,
#     }
#     _page_number = 1
#     while True:
#         resp = session.get(base_url + quote(json.dumps(variables)))
#         data = resp.json()
#         posts = data["data"]["user"]["edge_owner_to_timeline_media"]
#         for post in posts["edges"]:
#             yield post
#         page_info = posts["page_info"]
#         if _page_number == 1:
#             print(f"scraping total {posts['count']} posts of {user_id}")
#         else:
#             print(f"scraping page {_page_number}")
#         if not page_info["has_next_page"]:
#             break
#         if variables["after"] == page_info["end_cursor"]:
#             break
#         variables["after"] = page_info["end_cursor"]
#         _page_number += 1

# # Example run:
# if __name__ == "__main__":
#     with httpx.Client(
#         headers={
#             "x-ig-app-id": "936619743392459",
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
#             "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
#             "Accept-Encoding": "gzip, deflate, br",
#             "Accept": "*/*",
#         },
#         timeout=httpx.Timeout(20.0)
#     ) as client:
#         posts = list(scrape_user_posts("20800004785", client, page_limit=3))
#         print(json.dumps(posts, indent=2, ensure_ascii=False))
