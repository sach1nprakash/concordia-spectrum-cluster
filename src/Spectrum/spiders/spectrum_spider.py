'''
Main spider file, requires command line argument for the upper bound of pdf downloads
Use the command scrapy crawl spectrum_spider -s PDF_SAVE_LIMIT=100 
Once the PDF_SAVE_LIMIT is reached, statistics for the downloaded file are compiled
and the inverted index is built. 
'''
import os
import csv
import io
import scrapy
from scrapy import signals
from scrapy.signalmanager import dispatcher
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from PyPDF2 import PdfReader
from Spectrum.util import tokenizer, positional_index, pdf_statistics

class ConcordiaLibrarySpider(CrawlSpider):
    """
    Spider to crawl the Concordia University Spectrum Library for PDF files.
    Extracts metadata, verifies PDF content, and starts to build the inverted index.
    """

    name = 'spectrum_spider'

    # Start URL
    start_urls = ['https://spectrum.library.concordia.ca/']

    # Allowed domain
    allowed_domains = ['library.concordia.ca']

    # Custom settings for Scrapy
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
    }

    # Define link extraction rules
    rules = (
        Rule(LinkExtractor(allow=()), callback='parse_page', follow=True),
    )

    # Class variables
    downloaded_pdf_count = 0
    max_pdf_downloads = 0
    base_save_dir = '../pdf_downloads'
    stats_dir = '../stats'
    debug_log_path = '../stats/debug_log.csv'

    def __init__(self, *args, **kwargs):
        """
        Initialize the spider and ensure the debug log file is created with headers.
        """
        super().__init__(*args, **kwargs)
        dispatcher.connect(self.on_spider_closed, signal=signals.spider_closed)

        if not os.path.exists(self.stats_dir):
            os.mkdir(self.stats_dir)

        # Create debug log file with headers if it doesn't exist
        if not os.path.exists(self.debug_log_path):
            with open(self.debug_log_path, 'w', newline='', encoding='utf-8') as debug_log:
                writer = csv.writer(debug_log)
                writer.writerow(['PDF URL', 'Source Page', 'Status', 'Comment'])

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        """
        Initializes the spider with custom settings.
        """
        spider = super(ConcordiaLibrarySpider, cls).from_crawler(crawler, *args, **kwargs)
        spider.max_pdf_downloads = int(crawler.settings.get('PDF_SAVE_LIMIT', 0))
        return spider

    def parse_page(self, response):
        """
        Parse a page to extract PDF links and associated division metadata.
        """
        pdf_links = response.css('a.ep_document_link')

        for link in pdf_links:
            pdf_url = link.attrib.get('href')

            if pdf_url and pdf_url.endswith('.pdf'):
                save_dir = self.base_save_dir

                # Ensure directory exists
                os.makedirs(self.base_save_dir, exist_ok=True)

                # Construct absolute URL for the PDF
                full_pdf_url = response.urljoin(pdf_url)
                yield scrapy.Request(
                    full_pdf_url,
                    callback=self.process_pdf,
                    cb_kwargs={
                        'save_dir': save_dir,
                        'source_page_url': response.url
                    }
                )

    def process_pdf(self, response, save_dir, source_page_url):
        """
        Validate and save PDF files, logging the result.
        """
        pdf_url = response.url
        status = "Invalid"
        comment = ""

        if self.downloaded_pdf_count >= self.max_pdf_downloads:
            self.crawler.engine.close_spider(self, 'PDF_SAVE_LIMIT reached')
            return

        #Verify the headers for a pdf response
        if 'application/pdf' in response.headers.get('Content-Type', b'').decode('utf-8'):
            try:
                # Verify if the PDF has readable text
                pdf_reader = PdfReader(io.BytesIO(response.body))
                has_text = any(page.extract_text() for page in pdf_reader.pages)

                if has_text:
                    status = "Valid"
                    comment = "PDF contains readable text"

                    # Save the PDF if within the limit
                    if self.downloaded_pdf_count < self.max_pdf_downloads:
                        current_file = self.save_pdf(response.body, save_dir, pdf_url)
                        self.downloaded_pdf_count += 1

                        # Tokenize the saved PDF
                        tokenizer.tokenize_pdf_file(current_file)
                    else:
                        comment = "PDF not saved, download limit reached"
                else:
                    comment = "PDF does not contain readable text (scanned without OCR)"
            except Exception as e:
                comment = f"Error processing PDF: {e}"
        else:
            comment = "Content is not a PDF (may require a login)"

        self.log_debug_info(pdf_url, source_page_url, status, comment)
        yield {
            'url': pdf_url,
            'source_page': source_page_url,
            'status': status,
            'comment': comment
        }

    def save_pdf(self, content, save_dir, pdf_url):
        """
        Save the PDF content to disk.
        """
        file_name = f"{self.downloaded_pdf_count + 1}_{os.path.basename(pdf_url)}"
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, 'wb') as pdf_file:
            pdf_file.write(content)
        return file_path

    def log_debug_info(self, pdf_url, source_page_url, status, comment):
        """
        Log debug information to the CSV file.
        """
        with open(self.debug_log_path, 'a', newline='', encoding='utf-8') as debug_log:
            writer = csv.writer(debug_log)
            writer.writerow([pdf_url, source_page_url, status, comment])


    def on_spider_closed(self, spider):
        '''
        Triggers index generation and compiles statistics once the 
        spider closes
        '''
        print("Spider has closed. Starting positional indexing...")
        positional_index.init_pos_indexing()
        positional_index.construct_pos_index()
        print("Gathering statistics for downloaded pdf(s)...")
        pdf_statistics.main()
        