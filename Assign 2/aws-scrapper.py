import requests
from bs4 import BeautifulSoup
import re


import matplotlib.pyplot as plt

def scraper():
  # There are so few books published here is all of them, in a situation like this its best to use direct links as Amazon specifically prohibits use of web-crawlers but not web-scrappers which can be used in this isntance.
    urlList = [
      "https://www.amazon.com/Algorithm-Design-Manual-Computer-Science/dp/3030542556/ref=sr_1_1?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-1&ufe=app_do%3Aamzn1.fos.18ed3cb5-28d5-4975-8bc7-93deae8f9840",
      "https://www.amazon.com/Algorithm-Design-Manual-Steven-Skiena/dp/1848000693/ref=sr_1_2?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-2",
      "https://www.amazon.com/Calculated-Bets-Computers-Gambling-Mathematical/dp/0521009626/ref=sr_1_4?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-4",
      "https://www.amazon.com/Science-Design-Manual-Texts-Computer/dp/3319554433/ref=sr_1_3?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-3",
      "https://www.amazon.com/Implementing-Discrete-Mathematics-Combinatorics-Mathematica/dp/0201509431/ref=sr_1_13?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-13",
      "https://www.amazon.com/Algorithm-Design-Manual-Steven-Skiena/dp/1848000707/ref=sr_1_9?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-9",
      "https://www.amazon.com/Programming-Challenges-Contest-Training-Computer/dp/0387001638/ref=sr_1_5?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-5&ufe=app_do%3Aamzn1.fos.18ed3cb5-28d5-4975-8bc7-93deae8f9840",
      "https://www.amazon.com/Programming-Challenges-Contest-Training-Manual/dp/147578970X/ref=sr_1_20?qid=1699481255&refinements=p_27%3ASteven+Skiena&s=books&sr=1-20",
      "https://www.amazon.com/Desaf%C3%ADos-programaci%C3%B3n-entrenamiento-concursos-Spanish/dp/8412238044/ref=sr_1_6?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-6",
      "https://www.amazon.com/Dont-Stop-12-Multiplication-Over-Achievers/dp/B09WPW79SN/ref=sr_1_11?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-11",
      "https://www.amazon.com/Computational-Discrete-Mathematics-Combinatorics-Mathematica/dp/0521806860/ref=sr_1_10?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-10",
      "https://www.amazon.com/Whos-Bigger-Steven-Skiena-ebook/dp/B00EZ3VF34/ref=sr_1_7?qid=1699481217&refinements=p_27%3ASteven+Skiena&s=books&sr=1-7",
    ]
    # These headers allow us to scrape the page, may need to be periodically updates as Amazon becomes aware of working headers and blocks them
    HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
  'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.5',
  'Accept-Encoding':'gzip',
  'DNT':'1',
  'Connection':'close'
    }


    allBookInfoList = []

    for i in range(len(urlList)):
      # Get the page
      html = requests.get(urlList[i], headers=HEADERS)

      curBookInfo = []

      # Get each attribute
      soup = BeautifulSoup(html.content, "lxml")
      #print(soup)

      # Get the title and ISBN information
      title = soup.find('span', attrs={'id':"productTitle"}).string.strip()
      ISBN = soup.find('div', attrs={'id':"rpi-attribute-book_details-isbn13"}).find_all('span')[-1].string.strip()

      # Get the best seller ranking
      # This line is terrible, but seller position has no feature of ID or class, or even a tag associated
      # Contact amazon about their bad programming skills
      bestSellerRank = soup.find('div', attrs={'id': 'detailBulletsWrapper_feature_div'}).find_all('ul')[1].find_all('span', class_='a-list-item')[0].find('span', class_='a-text-bold').next_sibling.string
      bestSellerRank = re.findall('\d+', bestSellerRank)
      bestSellerRank = int(''.join(bestSellerRank))

      curBookInfo.append(title)
      curBookInfo.append(ISBN)
      curBookInfo.append(bestSellerRank)



      allBookInfoList.append(curBookInfo)

    return allBookInfoList

def visualize(allBookInfoList):
  # Extract book titles and their corresponding rank
  book_titles = [book[0] for book in allBookInfoList]
  rank = [book[2] for book in allBookInfoList]

  # Invert the ranked data
  inverted_rank = rank[::-1]

  # Create a bar chart
  plt.figure(figsize=(10, 6))
  plt.barh(book_titles, inverted_rank, color='skyblue')
  plt.xlabel('Ranking')
  plt.title('Book Ranking')

  # Adjust layout for better visibility of long book titles
  plt.tight_layout()

  # Show the bar graph
  plt.show()

  # Create a scatter plot
  plt.figure(figsize=(10, 6))
  plt.scatter(inverted_rank, range(len(book_titles)), c='skyblue', marker='o')
  plt.yticks(range(len(book_titles)), book_titles)
  plt.xlabel('Rank')
  plt.title('Book Rank (Scatter Plot)')

  # Adjust layout for better visibility of book titles
  plt.tight_layout()

  # Show the scatter plot
  plt.show()


if __name__ == "__main__":   
    
    bookArr = scraper()
    visualize(bookArr)

   