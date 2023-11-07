import requests
from bs4 import BeautifulSoup


if __name__ == "__main__":    
    url = "https://www.amazon.com/Algorithm-Design-Manual-Computer-Science/dp/3030542556/ref=sr_1_2?crid=SEBU0DFZSAD&keywords=steven+s.+skiena&qid=1699371429&sprefix=steven+s.+sk%2Caps%2C86&sr=8-2&ufe=app_do%3Aamzn1.fos.18ed3cb5-28d5-4975-8bc7-93deae8f9840"

    HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',
  'Accept-Language': 'en-US,en;q=0.5'
    }

    html = requests.get(url, headers=HEADERS)
    #print(html.text)
    soup = BeautifulSoup(html.content, "lxml")
    title = soup.find('span', attrs={'id':"productTitle"}).string.strip()
    #ISBN = soup.find('div', attrs={'id':"rpi-attribute-book_details-isbn13"}).find_all('span')[-1].string.strip()
    bestSellerRank = soup.find('div', attrs={'id':"detailBulletsWrapper_feature_div"}).find('span', class_='a-list-item')
    print(bestSellerRank)
    #price = soup.find('span', {'id':"price_inside_buybox"}).text.strip()
    #print(price)