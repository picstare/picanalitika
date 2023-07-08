import asyncio
import streamlit as st
from playwright.async_api import async_playwright
from datetime import datetime, timedelta
import dateparser

async def run_scraper(account_name, start_date, end_date):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(f"https://www.facebook.com/{account_name}")

        try:
            await page.get_by_role("dialog").locator("div").filter(has_text="Connect with").first.click()
            await page.get_by_role("button", name="Close").click()
        except:
            pass  # Dialog not found or encountered an error

        # Get the page title text
        title_element = await page.query_selector("h1")
        title_text = await title_element.inner_text()

        # Click on the page heading
        await page.get_by_role("heading", name=title_text).first.click()

        scroll_delay = 2  # Delay in seconds between scrolls
        current_date = start_date

        while current_date <= end_date:
            # Scroll to load more posts
            post_elements = await page.query_selector_all('div[role="article"]')
            await post_elements[-1].scroll_into_view_if_needed()

            # Wait for new items to load
            await page.wait_for_timeout(2000)

            # Check if new items were loaded
            items_on_page_after_scroll = len(await page.query_selector_all('div[role="article"]'))
            if items_on_page_after_scroll <= len(post_elements):
                break  # Stop scrolling if no more items are loaded

            # Increment the current_date by one day
            current_date += timedelta(days=1)

            # Wait for the specified scroll_delay
            await page.wait_for_timeout(scroll_delay * 1000)

        # Extract posts
        posts_locator = '.x1a2a7pz[role="article"]'
        posts = await page.query_selector_all(posts_locator)
        current_datetime = datetime.now()

        post_texts = []
        post_tags = []
        post_dates = []
        user_names = []
        image_attachments = []
        video_attachments = []
        like_counts = []
        love_counts = []
        comment_counts = []
        share_counts = []
        post_comments = []

        

        for post in posts:
            post_text_element = await post.query_selector('.x1iorvi4.x1pi30zi.x1l90r2v.x1swvt13')
            if post_text_element:
                see_more_button = await post_text_element.query_selector('div[role="button"]')
                if see_more_button:
                    # Click the "See more" button
                    await page.evaluate('(element) => element.click()', see_more_button)

                post_text = await post_text_element.inner_text()
                post_texts.append(post_text)
            else:
                post_texts.append("N/A")

            post_tags_element = await post.query_selector_all('.x1i10hfl.xjbqb8w.x6umtig.x1b1mbwd.xaqea5y.xav7gou.x9f619.x1ypdohk.xt0psk2.xe8uvvx.xdj266r.x11i5rnm.xat24cr.x1mh8g0r.xexx8yu.x4uap5.x18d9i69.xkhd6sd.x16tdsg8.x1hl2dhg.xggy1nq.x1a2a7pz.xt0b8zv.x1qq9wsj.xo1l8bm')
            hashtags = [await tag_element.inner_text() for tag_element in post_tags_element]
            post_tags.append(hashtags)

            post_date_element = await post.query_selector('.x4k7w5x.x1h91t0o.x1h9r5lt.x1jfb8zj.xv2umb2.x1beo9mf.xaigb6o.x12ejxvf.x3igimt.xarpa2k.xedcshv.x1lytzrv.x1t2pt76.x7ja8zs.x1qrby5j')
            post_date = await post_date_element.inner_text() if post_date_element else "N/A"

            if post_date == "N/A":
                # Handle the case where post_date is not found
                # You can assign a default value or handle it differently based on your requirements
                post_date = "Not available"

            elif post_date.endswith("h"):
                hours_ago = int(post_date.split()[0])
                post_date = current_datetime - timedelta(hours=hours_ago)
                post_date = post_date.strftime("%Y-%m-%d %H:%M:%S")  # Format as desired, e.g., "%Y-%m-%d %H:%M"

            elif post_date.endswith("w"):
                weeks_ago = int(post_date.split()[0])
                post_date = current_datetime - timedelta(weeks=weeks_ago)
                post_date = post_date.strftime("%Y-%m-%d %H:%M:%S")

            elif post_date.endswith("d"):
                days_ago = int(post_date.split()[0])
                post_date = current_datetime - timedelta(days=days_ago)
                post_date = post_date.strftime("%Y-%m-%d %H:%M:%S")

            elif post_date.endswith("m"):
                minutes_ago = int(post_date.split()[0])
                post_date = current_datetime - timedelta(minutes=minutes_ago)
                post_date = post_date.strftime("%Y-%m-%d %H:%M:%S")

            else:
                # Parse the date using dateparser
                parsed_date = dateparser.parse(post_date)

                if parsed_date is not None:
                    # Adjust the year if necessary (e.g., if the post is from last year)
                    if parsed_date.year > current_datetime.year or (parsed_date.year == current_datetime.year and parsed_date.month > current_datetime.month):
                        parsed_date = parsed_date.replace(year=current_datetime.year - 1)

                    post_date = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # Handle the case where the date parsing failed
                    post_date = "Invalid date"

            post_dates.append(post_date)

            user_name_element = await post.query_selector('.x1heor9g.x1qlqyl8.x1pd3egz.x1a2a7pz.x1gslohp.x1yc453h span')
            user_name = await user_name_element.inner_text() if user_name_element else "N/A"
            user_names.append(user_name)

            image_attachment_element = await post.query_selector('.x1n2onr6 img')
            image_attachment = await image_attachment_element.get_attribute('src') if image_attachment_element else "N/A"
            image_attachments.append(image_attachment)

            video_attachment_element = await post.query_selector('.x1n2onr6 video')
            video_attachment = await video_attachment_element.get_attribute('src') if video_attachment_element else "N/A"
            video_attachments.append(video_attachment)

            like_count_element = await post.query_selector('div.x1i10hfl[aria-label^="Like:"]')
            like_count = await like_count_element.get_attribute('aria-label') if like_count_element else "N/A"
            like_counts.append(like_count)

            love_count_element = await post.query_selector('div.x1i10hfl[aria-label^="Love:"]')
            love_count = await love_count_element.get_attribute('aria-label') if love_count_element else "N/A"
            love_counts.append(love_count)

            share_count_element = await post.query_selector('.x168nmei > div > div > div > div > div > div:nth-child(3) > div') or await post.query_selector('.x168nmei > div > div > div > div > div > div:nth-child(2) > div:nth-child(2)')
            share_count = await share_count_element.inner_text() if share_count_element else "N/A"
            share_counts.append(share_count)

            comment_count_element = await post.query_selector('.x168nmei > div > div > div > div > div > div:nth-child(2) > div>span>div>div>div')
            comment_count = await comment_count_element.inner_text() if comment_count_element else "N/A"
            comment_counts.append(comment_count)

            post_comment_element = await post.query_selector('.x168nmei.x13lgxp2.x30kzoy.x9jhf4c.x6ikm8r.x10wlt62> div > div.xzueoph') #new24062
            #.x1jx94hy.x12nagc
            
            if post_comment_element:
                view_more_button = await post_comment_element.query_selector("div[role='button'].x1i10hfl > span > span") 

                if view_more_button:
                    await page.evaluate('(element) => element.click()', view_more_button)
                    
                    await page.wait_for_timeout(1000)  

                post_comment_list = await post_comment_element.query_selector_all('div[role="article"][aria-label^="Comment"]')
                comments = []

                for post_comment in post_comment_list:
                    user_name_element = await post_comment.query_selector('.x1y1aw1k.xn6708d.xwib8y2.x1ye3gou > span.xt0psk2')
                    comment_text_element = await post_comment.query_selector('.x1lliihq.xjkvuk6.x1iorvi4')

                    if comment_text_element:
                        see_more_button = await comment_text_element.query_selector('div[role="button"]')
                        if see_more_button:
                            # Click the "See more" button
                            await page.evaluate('(element) => element.click()', see_more_button)

                        user_name = await user_name_element.inner_text() if user_name_element else "N/A"
                        comment_text = await comment_text_element.inner_text() if comment_text_element else "N/A"

                        comments.append({
                            'username': user_name,
                            'comment_text': comment_text
                        })

                post_comments.append(comments)
            else:
                post_comments.append([])

        await context.close()
        await browser.close()

        posts_data = []
        min_length = min(len(post_texts), len(user_names), len(post_dates), len(image_attachments), len(video_attachments), len(post_tags), len(like_counts), len(love_counts), len(comment_counts), len(share_counts), len(post_comments))
        

        for i in range(min_length):
            if post_dates[i].endswith("h"):
                hours_ago = int(post_dates[i].split()[0])
                post_date = current_datetime - timedelta(hours=hours_ago)
            elif post_dates[i].endswith("w"):
                weeks_ago = int(post_dates[i].split()[0])
                post_date = current_datetime - timedelta(weeks=weeks_ago)
            elif post_dates[i].endswith("d"):
                days_ago = int(post_dates[i].split()[0])
                post_date = current_datetime - timedelta(days=days_ago)
            elif post_dates[i].endswith("m"):
                minutes_ago = int(post_dates[i].split()[0])
                post_date = current_datetime - timedelta(minutes=minutes_ago)
            else:
                try:
                    post_date = datetime.strptime(post_dates[i], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    st.warning(f"Invalid date format: {post_dates[i]}")

            if start_date <= post_date.date() <= end_date:
                post_data = {
                    "username": user_names[i],
                    "date post": post_dates[i],
                    "post text": post_texts[i],
                    "image attachment": image_attachments[i],
                    "video attachment": video_attachments[i],
                    "hashtags": post_tags[i],
                    "like count": like_counts[i],
                    "love count": love_counts[i],
                    "comment count": comment_counts[i],
                    "share count": share_counts[i],
                    "comments": post_comments[i]
                }
                posts_data.append(post_data)

        return posts_data

        


async def main():
    await run_scraper("account_name", "start_date", "end_date")

if __name__ == "__main__":
    asyncio.run(main())