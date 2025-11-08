import pandas as pd
import email
from email.policy import default

# Load dataset (change path if needed)
df = pd.read_csv("/Users/nikitagrover/ml+proj/data/raw/emails.csv")

# Prepare a list to store cleaned emails
parsed_emails = []

for idx, row in df.iterrows():
    raw_msg = row['message']
    
    try:
        # Parse email using the standard library
        msg = email.message_from_string(raw_msg, policy=default)
        
        # Extract fields
        message_id = msg.get('Message-ID')
        date = msg.get('Date')
        sender = msg.get('From')
        recipient = msg.get('To')
        subject = msg.get('Subject')
        
        # Get the body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True)
            if body:
                body = body.decode(errors="ignore")
            else:
                body = ""
        
        parsed_emails.append({
            "file": row['file'],
            "Message-ID": message_id,
            "Date": date,
            "From": sender,
            "To": recipient,
            "Subject": subject,
            "Body": body.strip()
        })
        
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")

# Save clean dataset
clean_df = pd.DataFrame(parsed_emails)
clean_df.to_csv("/Users/nikitagrover/ml+proj/data/processed/emails_clean.csv", index=False)

print("✅ Parsing complete. Saved to clean_emails.csv")
