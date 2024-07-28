from time import sleep
import requests
from bs4 import BeautifulSoup
import csv

# from datetime import datetime
# Example with the standard date and time format
# date_str = '6 June 2019'
# date_format = '%d %B %Y' 

id = 'tt0455944' # The Equalizer - ~750 reviews For testing
# id = 'tt15398776' # Oppenheimer - 4K+ reviews supplementary data
# id = 'tt0468569' # The Dark Knight - 9K+ reviews supplementary data
# id = 'tt0111161' # The Shawshank Redemption - 11K+ reviews, for training
start_url = 'https://www.imdb.com/title/'+id+'/reviews?ref_=tt_urv'
link = 'https://www.imdb.com/title/'+id+'/reviews/_ajax'

params = {
    'ref_': 'undefined',
    'paginationKey': ''
}

r_n = []
with requests.Session() as s:
    s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
    res = s.get(start_url)

    while True:
        soup = BeautifulSoup(res.text,"lxml")
        for item in soup.select(".review-container"):
            # Scrape
            title = item.select_one("a.title").get_text(strip=True)
            author = item.select_one("span.display-name-link > a").get_text(strip=True)
            date = item.select_one("span.review-date").get_text(strip=True)
            stars = int(item.select_one("span.rating-other-user-rating > span").get_text()) if item.select_one("span.rating-other-user-rating > span") else None
            review = item.select_one("div.show-more__control").get_text(" ")
            permalink = item.select_one("div.text-muted > a")['href']

            # Process reviews
            review_cleaned = review.replace(',', ' ')

            # Poulate Review Object
            review = {
                'Title':title,
                'Author':author,
                'Date':date,
                'Stars(out_of_10)':stars,
                'Review':review_cleaned
            }
            r_n.append(review)

        try:
            pagination_key = soup.select_one(".load-more-data[data-key]").get("data-key")
        except AttributeError:
            break
        params['paginationKey'] = pagination_key
        res = s.get(link,params=params)
        sleep(1)

# Open the CSV file for writing in 'w' (write) mode
with open('reviews_'+id+'.csv', 'w', newline='', encoding='utf-8') as csvfile:
  # Create a CSV writer object
  writer = csv.writer(csvfile)

  # Write the header row manually
  header_row = r_n[0].keys()
  writer.writerow(header_row)

  # Write each dictionary as a row, converting values to strings
  for item in r_n:
    row_data = [str(value) for value in item.values()]  # Convert values to strings
    writer.writerow(row_data)