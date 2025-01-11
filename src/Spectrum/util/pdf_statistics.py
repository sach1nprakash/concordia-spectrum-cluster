'''
Compiles statistics for the downloaded pdf files. Uses beautifulsoup to 
parse the data and dataframes to efficiently manage the data. Can be 
used to compare the results of clustering. Needs the log file generated during crawling to 
get the statistics. Use command python pdf_statistics.py to run as a standalone file.
'''
import os
from collections import Counter
import requests
import pandas as pd
from bs4 import BeautifulSoup
from beautifultable import BeautifulTable


def fetch_page_content(url):
    """
    Fetch the HTML content of a given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None


def extract_department_and_school(soup):
    """
    Extract the department and school from the "Divisions:" row of the page.
    """
    department = "Unknown"
    school = "Unknown"
    division_row = soup.find('th', string="Divisions:")

    if division_row:
        division_cell = division_row.find_next_sibling('td', class_="ep_row")
        if division_cell:
            links = division_cell.find_all('a')
            if links:
                department = links[-1].text.strip()
                if len(links) == 2 and links[-1].text.strip() == "Library":
                    school = "Library"
                elif len(links) > 1:
                    school = links[-2].text.strip()
    return department, school


def extract_thesis_supervisors(soup):
    """
    Extract thesis supervisors from the meta tag.
    """
    return soup.find('meta', {'name': 'eprints.thesis_advisors_name'})['content'] if soup.find('meta', {'name': 'eprints.thesis_advisors_name'}) else "Unknown"


def extract_university(soup):
    """
    Extract the university from the meta tag.
    """
    return soup.find('meta', {'name': 'eprints.institution'})['content'] if soup.find('meta', {'name': 'eprints.institution'}) else "Unknown"


def extract_page_details(url):
    """
    Extract details from a given source page.
    """
    try:
        soup = fetch_page_content(url)
        if not soup:
            raise ValueError("Failed to fetch page content.")

        department, school = extract_department_and_school(soup)
        if department == "Unknown":
            program_row = soup.find('th', string="Program:")
            if program_row:
                program_cell = program_row.find_next_sibling(
                    'td', class_="ep_row")
                if program_cell:
                    department = program_cell.text.strip()

        supervisors = extract_thesis_supervisors(soup)
        university = extract_university(soup)

        return {
            "Department": department,
            "School": school,
            "University": university,
            "Thesis Supervisor(s)": supervisors,
        }
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return {
            "Department": "Error",
            "School": "Error",
            "University": "Error",
            "Thesis Supervisor(s)": "Error",
        }


def display_statistics_table(stats):
    """
    Display a summary table with key statistics.
    """
    table = BeautifulTable()
    table.columns.header = ["Category", "Details"]
    for category, counts in stats.items():
        if isinstance(counts, Counter):
            counts_summary = "\n".join(
                [f"{key}: {value}" for key, value in counts.items()])
            table.rows.append([category, counts_summary])
        else:
            table.rows.append([category, counts])
    return str(table)


def process_data(input_file, summary_file):
    """
    Process the input data and save results and statistics.
    """
    data = pd.read_csv(input_file)
    processed_data = []

    for _, row in data.iterrows():
        source_url = row["Source Page"]
        details = extract_page_details(source_url)
        processed_data.append({
            "PDF URL": row["PDF URL"],
            "Source Page": source_url,
            "Status": row["Status"],
            "Comment": row["Comment"],
            **details,
        })

    output_df = pd.DataFrame(processed_data)
    # output_df.to_csv(output_file, index=False)

    statistics = {
        "Total Entries": len(output_df),
        "University Counts": Counter(output_df["University"]),
        "Department Counts": Counter(output_df["Department"]),
        "School Counts": Counter(output_df["School"]),
        "Supervisor Counts": Counter(output_df["Thesis Supervisor(s)"]),
    }

    with open(summary_file, "w", encoding='utf-8') as file:
        file.write("Summary:\n")
        file.write(display_statistics_table(statistics))

    print(f"Summary saved to {summary_file}")


def main():
    '''
    Main method
    '''
    cwd = os.path.dirname(os.path.realpath(__file__))
    root = os.path.dirname(os.path.realpath(cwd))

    output_directory = os.path.join(root, 'stats')
    input_file = os.path.join(output_directory, 'debug_log.csv')
    # output_file = os.path.join(
    #     output_directory, 'output_with_school_and_library.csv')
    summary_file = os.path.join(output_directory, 'summary.txt')
    process_data(input_file, summary_file)


if __name__ == "__main__":
    main()
