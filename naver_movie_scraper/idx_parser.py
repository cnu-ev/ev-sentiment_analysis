from bs4 import BeautifulSoup
import re
import sys
import io
from urllib.request import urlopen


def make_a_tag_href_attribute_list (page_numbers) :
    search_url = "https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=pnt&date=20190708&page="+str(page_numbers)
    # search_url = "https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=pnt&date=20190708&page=1"
    html = urlopen(search_url)
    html_contents = str(html.read().decode('euc-kr'))
    html_contents
    a_tag_href_attribute_list = re.findall("/movie/bi/mi/basic.nhn\?code=\d\d\d\d\d\d*",html_contents)
    a_tag_href_attribute_list
    return a_tag_href_attribute_list

def extract_idx_from_a_tag_href_attribute_list (a_tag_href_attribute_list):
    idx_list = []
    for a_tag_href_attribute in a_tag_href_attribute_list:
        idx_list +=  [   re.findall("\d\d\d\d\d\d*", a_tag_href_attribute)[0] ]
    return idx_list
