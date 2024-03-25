from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
import io

# Define the scopes required for the Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

def service_account_login():
    """
    Authenticate and create a service object for Google Drive API.
    Returns:
        service: Authenticated Google Drive API service.
    """
    # Load the OAuth2 credentials from the client secrets file
    flow = InstalledAppFlow.from_client_secrets_file(
        'path_to_your_client_secret_json_file', SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)

def download_file(service, file_id, file_path):
    """
    Download a file from Google Drive.
    Parameters:
        service: Authenticated Google Drive service.
        file_id (str): ID of the file to be downloaded.
        file_path (str): Local path to save the downloaded file.
    """
    # Request to download the file
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download of {file_path} {int(status.progress() * 100)}% complete.")

# Authenticate and create the Google Drive API service
service = service_account_login()

# File IDs and paths (replace with your own paths and file IDs)
tax_source_id = '1X4dAHsPFabEE4D0HX9O2O1gbaBGBLlxt'
tax_related_id = '1gUhbxmk213Z4PQugyr0UyBSVFBR1qkAv'
tax_source_path = 'path_where_you_want_to_save_tax_source_file'
tax_related_path = 'path_where_you_want_to_save_tax_related_file'

# Download the files
download_file(service, tax_source_id, tax_source_path)
download_file(service, tax_related_id, tax_related_path)
