from pydantic import BaseModel
class BookCreate(BaseModel):
 title: str
 author: str

class Book(BookCreate):
     id: int

import sqlite3

def create_connection():
 connection = sqlite3.connect("books.db")
 return connection

def create_table():
 connection = create_connection()
 cursor = connection.cursor()
 cursor.execute("""
 CREATE TABLE IF NOT EXISTS books (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 title TEXT NOT NULL,
 author TEXT NOT NULL
 )
 """)
 connection.commit()
 connection.close()

create_table() # Call this function to create the table
def create_book(book: BookCreate):
 connection = create_connection()
 cursor = connection.cursor()
 cursor.execute("INSERT INTO books (title, author) VALUES (?, ?)", (book.title, book.author))
 connection.commit()
 connection.close()

@app.post("/books/")
def create_book_endpoint(book: BookCreate):
 book_id = create_book(book)
 return {"id": book_id, **book.dict()}
@app.post("/books/")
def create_book(book: BookCreate):
 # Logic to add the book to the database
 return book