import numpy as np
import requests
import lxml.html as html
import matplotlib.pyplot as plt


r = requests.get("https://www.kaggle.com/c/predict-west-nile-virus/leaderboard")
data = r.text
table = doc.xpath("//table")
table = table[0]
rows = table.xpath("//tr")


def process_rows(rows):
    score = np.zeros(len(rows) -1)
    entries = np.zeros(len(rows) -1)

    for i, row in enumerate(rows[1:]):
        cells = row.getchildren()
        try:
            score[i] = float(cells[3].text_content().strip())
            try:
                entries[i] = float(cells[4].text_content().strip())
            except:
                entries[i] = 1
        except:
            print(i,
                  "Cells:",
                  cells[3].text_content(),
                  cells[4].text_content())
            raise
    return score, entries

s, e = process_rows(rows)
plt.scatter(e, s, alpha=0.25)
plt.show()

# plt.clf()

    
