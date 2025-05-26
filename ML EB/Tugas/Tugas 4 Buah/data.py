import pandas as pd
import requests
import re
import io
import os
import time
from urllib.parse import urlparse, parse_qs

def extract_links(combined_links):
    parts = re.split(r'(?=https://)', combined_links)
    links = [part for part in parts if part.startswith('https://')]
    return links

def download_csv_from_drive(file_id):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            except Exception as e:
                print(f"Error parsing CSV with first method: {e}")
        
        url = f"https://drive.google.com/file/d/{file_id}/view"
        print(f"Trying alternative method for {url}")
        
        session = requests.Session()
        response = session.get(url)
        
        if response.status_code == 200:
            try:
                export_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                response = session.get(export_url)
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            except Exception as e:
                print(f"Error with alternative method: {e}")
                return None
        else:
            print(f"Failed to download file with ID {file_id}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error during download: {e}")
        return None

def download_sheet_from_drive(sheet_id):
    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        response = requests.get(url)
        if response.status_code == 200:
            try:
                return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            except Exception as e:
                print(f"Error parsing sheet: {e}")
                return None
        else:
            print(f"Failed to download sheet with ID {sheet_id}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error during sheet download: {e}")
        return None

def extract_file_id(url):
    try:
        if 'drive.google.com/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
            return file_id
        elif 'docs.google.com/spreadsheets/d/' in url:
            sheet_id = url.split('/spreadsheets/d/')[1].split('/')[0]
            return sheet_id
        elif 'drive.google.com/uc?export=download&id=' in url:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            return query_params.get('id', [None])[0]
        else:
            print(f"Unsupported URL format: {url}")
            return None
    except Exception as e:
        print(f"Error extracting file ID from {url}: {e}")
        return None

def process_link(link):
    file_id = extract_file_id(link)
    if not file_id:
        return None
    
    if 'docs.google.com/spreadsheets' in link:
        return download_sheet_from_drive(file_id)
    else:
        return download_csv_from_drive(file_id)

def save_individual_csv(df, index, folder="individual_csvs"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = os.path.join(folder, f"source_{index}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved individual CSV to {filename}")

def main():
    combined_links = """https://drive.google.com/file/d/1i_9ka1G2nGsot-8uhAG5qGoTNG43IJje/view?usp=sharinghttps://drive.google.com/file/d/1YvSR3kFb-LBweOqC9zLQet8r1fgCAbEe/view?usp=sharinghttps://drive.google.com/file/d/1lhciDgpwpmiFM5g1PRjEDaY7qJFxLulk/view?usp=drive_linkhttps://drive.google.com/file/d/1EHb8TBtHslxcmqhWAQPZ4oaswlsuvUK0/view?usp=drive_linkhttps://drive.google.com/file/d/1ss__2m1UDGkIEQc_kvO1UahOZARDHio1/view?usp=sharinghttps://drive.google.com/file/d/1VJujEhQj0-bZMTNVN6IrJaRVKPbq67uA/view?usp=drive_linkhttps://drive.google.com/file/d/13nJGU4Vop613ktqiZ6mcyxdZRF-st-CN/view?usp=sharinghttps://docs.google.com/spreadsheets/d/1sJpf9maa-qDq82w8X7cTelmLOLfjlbvRG6qjFkPysVI/edit?usp=sharinghttps://docs.google.com/spreadsheets/d/1HJ6byfqNgr3cqUflmL-LAS9hc8Jwc-BcOq1zrOiO19c/edit?usp=sharinghttps://drive.google.com/file/d/1IPz7Cc1n4ByOqUyGf2JHz2Q8n3ZnUSgf/view?usp=sharinghttps://drive.google.com/file/d/1CCwkhQynj2Ii4weNjDjKrK53zD_7yh_f/view?usp=sharinghttps://docs.google.com/spreadsheets/d/1OuFAXzjY2Vosj6j3Sm3otFvKabdkZLl1/edit?usp=sharing&ouid=105011830416643335170&rtpof=true&sd=truehttps://drive.google.com/file/d/1DCUrPAsXnnfO9NLDCv8zp_Xqkx--dkcn/view?usp=drive_linkhttps://drive.google.com/file/d/10nK_vAmz5ALg4m9Z6rOM1r-K01sUVy8x/view?usp=sharinghttps://drive.google.com/file/d/1NmrE_lgSwpJ52evhdVy4W_4K0MAvpW3b/view?usp=sharinghttps://drive.google.com/file/d/1rmBXz0uaNd3MzMPbd_YFWcaR_yEwMQoR/view?usp=sharinghttps://drive.google.com/file/d/1UqXkpy5BKZxYEoy91jppZLXLzYIcyry_/view?usp=sharinghttps://drive.google.com/file/d/1Hs1Y1caaeRTd-Vy3rTsnD3IMTDH3Q0js/view?usp=sharinghttps://drive.google.com/file/d/1EM_lTewgsTubjn58mjVi4Mk8MTiqN2cA/view?usp=drive_linkhttps://drive.google.com/file/d/13N28_A6BOO4XepqvxE2uHfntfdKdtWSa/view?usp=drive_linkhttps://docs.google.com/spreadsheets/d/1gBNMXlQfcDtJK6pjmB5YbPRJk8jKGAVjC4pjBuRR-4Y/edit?usp=sharinghttps://drive.google.com/file/d/1hwUPo2_0d2GJ4p33AClu00PydlBtfzkR/view?usp=sharinghttps://drive.google.com/file/d/1pQZ1YWyDgdW75NeW-ytiU_yi9Iev7Sh8/view?usp=sharinghttps://drive.google.com/file/d/1Pf49LaypNOXMkQaqoscsYJ4gQ8h-82e8/view?usp=sharinghttps://docs.google.com/spreadsheets/d/1HJ6byfqNgr3cqUflmL-LAS9hc8Jwc-BcOq1zrOiO19c/edit?usp=sharinghttps://drive.google.com/file/d/1Le0aIDGbZEDDbV9Zr4pb4oQoCqgqMaDp/view?usp=sharinghttps://drive.google.com/file/d/1sH5F2tlt4BPO2j-TR2GNagFKd7HYkLSW/view?usp=drive_linkhttps://drive.google.com/file/d/16vIxIEN499AOmrEr3MA4-Tv6BeT_nLyr/view?usp=sharinghttps://drive.google.com/file/d/1pUYPN8_kVVGDEMYp0XBLaigulQY3uPVK/view?usp=sharinghttps://drive.google.com/file/d/1LW9vkLnyhfb_mYdFQ5qB3BbsOKLDaZx_/view?usp=sharinghttps://drive.google.com/file/d/1NoLtfCLUe2dAr3Ac0fxXqgwxLwqbtuhQ/view?usp=sharinghttps://drive.google.com/file/d/17FsW-1ltHhqDeQ9DkDNOYy00Bivy3lJS/view?usp=sharinghttps://drive.google.com/file/d/1_y9JG7_di_FFxZijp4qaF8PNyRfEGiOl/view?usp=sharinghttps://docs.google.com/spreadsheets/d/1vgObEQTl9RHftKTxjU2AlBigReK-WbZ3lo3OHt8bUeQ/edit?usp=sharinghttps://docs.google.com/spreadsheets/d/1Sj08vcbqQ9WIHAoLstwyz5Bz5k0YOJ53/edit?usp=sharing&ouid=117400838184951060020&rtpof=true&sd=truehttps://drive.google.com/file/d/1cNH1Kp4VTeVMyhBVRLrp6rnNl92aYs5J/view?usp=sharinghttps://drive.google.com/file/d/1UgUkP_LRjc77VKPHXP5eVERcmsseZ7fD/view?usp=sharinghttps://drive.google.com/file/d/1BgWnTZe8bLS6nYSFl12VHqyqmxa5L5W2/view?usp=drive_linkhttps://drive.google.com/file/d/1GI_tEx54RfmnL5c9uw7Pq1W89d-IEhjU/view?usp=drive_linkhttps://drive.google.com/file/d/18NXzTv64-T_ETdgaujvvoUABtK7DSeo_/view?usp=drive_linkhttps://drive.google.com/drive/folders/1S6SVn4lRdjbJ-3cNz8WzQgTBFcyWvzRr?usp=drive_linkhttps://drive.google.com/file/d/1LwaDTWPu7DTYQiJ3k0Y-P_a1yuNGBFbZ/view?usp=sharinghttps://docs.google.com/spreadsheets/d/1kPFQgEGFT7Tl7jISfJFbHSFVg8A0zorb7_g1Q2M6La0/edit?usp=drive_linkhttps://drive.google.com/file/d/1Gn56Yt9ML9M6H28qqquo9hk8vXxAOtG3/view?usp=sharing"""
    
    links = extract_links(combined_links)
    print(f"Found {len(links)} links")
    
    all_dataframes = []
    success_count = 0
    failure_count = 0
    
    for i, link in enumerate(links):
        print(f"\nProcessing link {i+1}/{len(links)}: {link}")
        
        if 'drive.google.com/drive/folders' in link:
            print("Skipping folder link")
            continue
        
        if i > 0 and i % 3 == 0:
            print("Pausing briefly to avoid rate limiting...")
            time.sleep(2)
        
        df = process_link(link)
        
        if df is not None:
            print(f"Successfully downloaded data. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            required_columns = ['Mean H', 'Mean S', 'Mean V', 'Label', 'Filename']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                failure_count += 1
                continue
            
            save_individual_csv(df, i)
            all_dataframes.append(df)
            success_count += 1
            print(f"Successfully processed dataset {i+1}")
        else:
            print(f"Failed to process link {i+1}")
            failure_count += 1
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nCombined data shape: {combined_df.shape}")
        
        combined_df.to_csv('fitur_buah.csv', index=False)
        print("Data saved to fitur_buah.csv")
        
        print(f"\nSummary:")
        print(f"Total links processed: {len(links)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")
        print(f"Total rows in combined dataset: {combined_df.shape[0]}")
        
        if 'Label' in combined_df.columns:
            label_counts = combined_df['Label'].value_counts()
            print("\nClass distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} samples ({count/combined_df.shape[0]*100:.1f}%)")
    else:
        print("\nNo valid data found. Please check the URLs and make sure the files are accessible.")
        print("Individual CSV files may still be available in the 'individual_csvs' directory for inspection.")

if __name__ == "__main__":
    main()