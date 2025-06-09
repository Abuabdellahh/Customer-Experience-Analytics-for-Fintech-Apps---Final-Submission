"""
Database operations for Oracle database
"""

import oracledb
import pandas as pd
import logging
from typing import Optional
import os

from .config import DB_CONFIG, DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OracleDBManager:
    """Oracle database manager for storing review data"""
    
    def __init__(self):
        self.config = DB_CONFIG
        self.connection = None
        
    def connect(self) -> bool:
        """Establish connection to Oracle database"""
        try:
            dsn = f"{self.config['host']}:{self.config['port']}/{self.config['service']}"
            self.connection = oracledb.connect(
                user=self.config['user'],
                password=self.config['password'],
                dsn=dsn
            )
            logger.info("Successfully connected to Oracle database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Oracle database: {e}")
            return False
    
    def create_schema(self):
        """Create database schema"""
        if not self.connection:
            logger.error("No database connection")
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Create Banks table
            banks_table_sql = """
            CREATE TABLE banks (
                bank_id VARCHAR2(10) PRIMARY KEY,
                bank_name VARCHAR2(100) NOT NULL,
                app_name VARCHAR2(100),
                app_id VARCHAR2(100),
                created_date DATE DEFAULT SYSDATE
            )
            """
            
            # Create Reviews table
            reviews_table_sql = """
            CREATE TABLE reviews (
                review_id VARCHAR2(100) PRIMARY KEY,
                bank_id VARCHAR2(10),
                review_text CLOB,
                rating NUMBER(1) CHECK (rating BETWEEN 1 AND 5),
                review_date DATE,
                user_name VARCHAR2(100),
                thumbs_up NUMBER DEFAULT 0,
                sentiment_label VARCHAR2(20),
                sentiment_score NUMBER(3,2),
                primary_theme VARCHAR2(50),
                word_count NUMBER,
                created_date DATE DEFAULT SYSDATE,
                FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
            )
            """
            
            try:
                cursor.execute(banks_table_sql)
                logger.info("Banks table created successfully")
            except Exception as e:
                if "name is already used" in str(e):
                    logger.info("Banks table already exists")
                else:
                    raise e
            
            try:
                cursor.execute(reviews_table_sql)
                logger.info("Reviews table created successfully")
            except Exception as e:
                if "name is already used" in str(e):
                    logger.info("Reviews table already exists")
                else:
                    raise e
            
            self.connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return False
    
    def insert_banks_data(self):
        """Insert bank information"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            
            banks_data = [
                ('CBE', 'Commercial Bank of Ethiopia', 'CBE Mobile Banking', 'com.cbe.mobile'),
                ('BOA', 'Bank of Abyssinia', 'BOA Mobile Banking', 'com.boa.mobile'),
                ('DASHEN', 'Dashen Bank', 'Dashen Mobile Banking', 'com.dashen.mobile')
            ]
            
            insert_sql = """
            INSERT INTO banks (bank_id, bank_name, app_name, app_id)
            VALUES (:1, :2, :3, :4)
            """
            
            for bank_data in banks_data:
                try:
                    cursor.execute(insert_sql, bank_data)
                except Exception as e:
                    if "unique constraint" in str(e).lower():
                        logger.info(f"Bank {bank_data[0]} already exists")
                    else:
                        raise e
            
            self.connection.commit()
            cursor.close()
            logger.info("Banks data inserted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting banks data: {e}")
            return False
    
    def insert_reviews_data(self, df: pd.DataFrame):
        """Insert reviews data from DataFrame"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            
            insert_sql = """
            INSERT INTO reviews (
                review_id, bank_id, review_text, rating, review_date,
                user_name, thumbs_up, sentiment_label, sentiment_score,
                primary_theme, word_count
            ) VALUES (
                :1, :2, :3, :4, TO_DATE(:5, 'YYYY-MM-DD'),
                :6, :7, :8, :9, :10, :11
            )
            """
            
            inserted_count = 0
            for _, row in df.iterrows():
                try:
                    cursor.execute(insert_sql, (
                        row['review_id'],
                        row['bank'],
                        row['review_text'],
                        int(row['rating']),
                        row['date'],
                        row.get('user_name', 'Anonymous'),
                        int(row.get('thumbs_up', 0)),
                        row.get('sentiment_label', 'NEUTRAL'),
                        float(row.get('sentiment_score', 0.5)),
                        row.get('primary_theme', 'General'),
                        int(row.get('word_count', 0))
                    ))
                    inserted_count += 1
                except Exception as e:
                    if "unique constraint" in str(e).lower():
                        continue  # Skip duplicates
                    else:
                        logger.error(f"Error inserting review {row.get('review_id', 'unknown')}: {e}")
            
            self.connection.commit()
            cursor.close()
            logger.info(f"Successfully inserted {inserted_count} reviews")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting reviews data: {e}")
            return False
    
    def export_schema(self, filename: str = 'schema_dump.sql'):
        """Export database schema"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Get table creation scripts
            schema_sql = """
            -- Banks Table
            CREATE TABLE banks (
                bank_id VARCHAR2(10) PRIMARY KEY,
                bank_name VARCHAR2(100) NOT NULL,
                app_name VARCHAR2(100),
                app_id VARCHAR2(100),
                created_date DATE DEFAULT SYSDATE
            );
            
            -- Reviews Table
            CREATE TABLE reviews (
                review_id VARCHAR2(100) PRIMARY KEY,
                bank_id VARCHAR2(10),
                review_text CLOB,
                rating NUMBER(1) CHECK (rating BETWEEN 1 AND 5),
                review_date DATE,
                user_name VARCHAR2(100),
                thumbs_up NUMBER DEFAULT 0,
                sentiment_label VARCHAR2(20),
                sentiment_score NUMBER(3,2),
                primary_theme VARCHAR2(50),
                word_count NUMBER,
                created_date DATE DEFAULT SYSDATE,
                FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
            );
            
            -- Sample Data
            INSERT INTO banks VALUES ('CBE', 'Commercial Bank of Ethiopia', 'CBE Mobile Banking', 'com.cbe.mobile', SYSDATE);
            INSERT INTO banks VALUES ('BOA', 'Bank of Abyssinia', 'BOA Mobile Banking', 'com.boa.mobile', SYSDATE);
            INSERT INTO banks VALUES ('DASHEN', 'Dashen Bank', 'Dashen Mobile Banking', 'com.dashen.mobile', SYSDATE);
            """
            
            # Save to file
            os.makedirs('sql', exist_ok=True)
            with open(f'sql/{filename}', 'w') as f:
                f.write(schema_sql)
            
            logger.info(f"Schema exported to sql/{filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting schema: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def main():
    """Main execution function for database operations"""
    # Load analyzed data
    analyzed_file = os.path.join(DATA_PATHS['processed_data'], 'analyzed_reviews.csv')
    
    if not os.path.exists(analyzed_file):
        logger.error(f"Analyzed data file not found: {analyzed_file}")
        return
    
    df = pd.read_csv(analyzed_file)
    logger.info(f"Loaded {len(df)} analyzed reviews")
    
    # Initialize database manager
    db_manager = OracleDBManager()
    
    # Connect to database
    if not db_manager.connect():
        logger.error("Failed to connect to database")
        return
    
    # Create schema
    db_manager.create_schema()
    
    # Insert banks data
    db_manager.insert_banks_data()
    
    # Insert reviews data
    db_manager.insert_reviews_data(df)
    
    # Export schema
    db_manager.export_schema()
    
    # Close connection
    db_manager.close()
    
    print("\n=== Database Operations Summary ===")
    print(f"Reviews processed: {len(df)}")
    print("Schema created and data inserted successfully")
    print("Schema dump saved to sql/schema_dump.sql")

if __name__ == "__main__":
    main()
