import sqlite3

def create_database():
    conn = sqlite3.connect("trading_bot.db")
    cursor = conn.cursor()

    # Create strategies table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        strategy_data TEXT NOT NULL,
        performance_metrics TEXT
    )
    """)

    # Create backtests table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS backtests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        metrics TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    print("Database and tables created successfully.")

if __name__ == "__main__":
    create_database()
