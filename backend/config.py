
# Centralized Configuration for PETA2 Backend
import os

# --- SNOWFLAKE CREDENTIALS ---
# Verified working with Native Python 3.10 Connector
SNOWFLAKE_CONFIG = {
    'account': 'COZENTUS-DATAPRACTICE',
    'user': 'HACKATHON_DT',
    # The User's JWT Token (passed as password)
    'password': 'eyJraWQiOiIxOTMxNTY4MzQxMDAzOTM3OCIsImFsZyI6IkVTMjU2In0.eyJwIjoiMjk0NzMzOTQwNDEzOjI5NDczMzk0MjUzMyIsImlzcyI6IlNGOjIwMTciLCJleHAiOjE3NzEyMjY3MTF9.O0OTFEyQPIqpdCsNuV881UG1RtQQLBMIyUt-0kfESVYaI0J_u3S4fysE7lee7lWMIMoezOhd2t7gUItdoHC0UA',
    'database': 'HACAKATHON',
    'schema': 'DT_INGESTION',
    'warehouse': 'cozentus_wh'
}
