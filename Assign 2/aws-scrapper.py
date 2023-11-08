import requests
from bs4 import BeautifulSoup
import re


if __name__ == "__main__":   
    # There are so few books this man published here is all of them 
    urlList = [
       "https://www.amazon.com/Algorithm-Design-Manual-Computer-Science/dp/3030542556/ref=sr_1_2?crid=SEBU0DFZSAD&keywords=steven+s.+skiena&qid=1699371429&sprefix=steven+s.+sk%2Caps%2C86&sr=8-2&ufe=app_do%3Aamzn1.fos.18ed3cb5-28d5-4975-8bc7-93deae8f9840",
        "https://www.amazon.com/Programming-Challenges-Contest-Training-Computer/dp/0387001638/ref=sr_1_2?keywords=steven+s.+skiena&qid=1699414063&sr=8-2&ufe=app_do%3Aamzn1.fos.18ed3cb5-28d5-4975-8bc7-93deae8f9840",
        
    ]
    HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
  'Accept-Language': 'en-US,en;q=0.5'
    }

    for i in range(len(urlList)):
      # Get the page
      html = requests.get(urlList[i], headers=HEADERS)

      # Get each attribute
      soup = BeautifulSoup(html.content, "lxml")
      title = soup.find('span', attrs={'id':"productTitle"}).string.strip()
      ISBN = soup.find('div', attrs={'id':"rpi-attribute-book_details-isbn13"}).find_all('span')[-1].string.strip()
      # This line is terrible, but seller position has no feature of ID or class, or even a tag associated
      # Contact amazon about their bad programming skills
      bestSellerRank = soup.find('div', attrs={'id': 'detailBulletsWrapper_feature_div'}).find_all('ul')[1].find_all('span', class_='a-list-item')[0].find('span', class_='a-text-bold').next_sibling.string
      bestSellerRank = re.findall('\d+', bestSellerRank)
      bestSellerRank = int(bestSellerRank[0] + bestSellerRank[1])

      print(title)
      print(ISBN)
      print(bestSellerRank)

      #price = soup.find('span', {'id':"price_inside_buybox"}).text.strip()
      #print(price)