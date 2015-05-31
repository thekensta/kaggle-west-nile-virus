"""score-to-attempts.py

Look at the relationship between number of entries and performance.

"""

import numpy as np
import requests
import lxml.html as html
import matplotlib.pyplot as plt

def process_rows(rows):
    """Process all the rows in the Leaderboard table extracting score and
    entries.
    """

    score = np.zeros(len(rows) -1)
    entries = np.zeros(len(rows) -1)

    for i, row in enumerate(rows[1:]):
        cells = row.getchildren()
        try:
            score[i] = float(cells[3].text_content().strip())
            try:
                # There is a benchmark row in the table that
                # doesn't have an entries row
                entries[i] = float(cells[4].text_content().strip())
            except:
                entries[i] = 1
        except:
            # Abort for other rows
            print(i,
                  "Cells:",
                  cells[3].text_content(),
                  cells[4].text_content())
            raise
    return score, entries

def main():
    """Show a scatter plot of entries vs score."""

    r = requests.get("https://www.kaggle.com/c/predict-west-nile-virus/leaderboard")
    data = r.text
    doc = html.fromstring(data)
    tables = doc.xpath("//table")
    table = tables[0]
    rows = table.xpath("//tr")
    s, e = process_rows(rows)
    plt.scatter(e, s, alpha=0.25)
    plt.show()


if __name__ == "__main__":
    main()

