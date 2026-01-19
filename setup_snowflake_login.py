"""
Test Snowflake Login Setup
Checks if ROLE_LOGIN table exists and creates sample user
"""
import snowflake.connector
from backend.config import SNOWFLAKE_CONFIG

def setup_login_table():
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'ROLE_LOGIN'")
        exists = cursor.fetchone()
        
        if not exists:
            print("Creating ROLE_LOGIN table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ROLE_LOGIN (
                    USERNAME VARCHAR(100),
                    PASSWORD VARCHAR(100),
                    USER_ROLE VARCHAR(50),
                    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            print("✅ Table created")
            
            # Insert default admin user
            cursor.execute("""
                INSERT INTO ROLE_LOGIN (USERNAME, PASSWORD, USER_ROLE)
                VALUES ('admin', 'admin', 'admin')
            """)
            print("✅ Default admin user created")
        else:
            print("✅ ROLE_LOGIN table already exists")
            
        # Check existing users
        cursor.execute("SELECT USERNAME, USER_ROLE FROM ROLE_LOGIN")
        users = cursor.fetchall()
        print(f"\nExisting users ({len(users)}):")
        for user in users:
            print(f"  - {user[0]} (role: {user[1]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SNOWFLAKE LOGIN TABLE SETUP")
    print("=" * 60)
    setup_login_table()
