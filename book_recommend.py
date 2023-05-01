import pandas as pd
import numpy as np
from joblib import load
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import urllib.request

#--------------Unpickling the pickled files-----------------

rating_table = load('rating_table.joblib')
books_image_data = load('books_image_data.joblib')

#----------------------Model fitting------------------------

sparse_matrix = csr_matrix(rating_table)
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_matrix)

#-------------------Creating recommeder function---------------

#This function takes a book name >><str> and returns a list of 5 books and their image link.
def recommend(book_name):
  """
  @param -- <str> book_name.
  @returns -- Two <list> of <str> 
  """
  recommended_books = []
  image_url = []
  #Extract the index of input book.
  book_index = np.where(rating_table.index==book_name)[0][0]
  distances , suggestions = model.kneighbors(rating_table.iloc[book_index,:].values.reshape(1,-1),n_neighbors=6)

  #Convert suggested 2d array into 1d array.
  suggestions = np.ravel(suggestions, order='C')

  #Get recommended books name.
  for i in suggestions:
    recommended_books.append(rating_table.index[i])

  #Get image link of those recommended books.
  for i in recommended_books:
    image_url.append(books_image_data[books_image_data["title"] == i ].image.to_string(index=False))
    
  return recommended_books,image_url
#---------------Implementing model into web page-----------------

#Refer streamlit documentation for frontend.

st.subheader("Collaborative Filtering Based Books Recommender Engine") #Title

#Extracting the books name from the loaded pickled rating table
books_name = rating_table.index.to_list()
#Dropdown select menu
selected_book = st.selectbox(
     'Search Your Book Here',
     books_name)
    


if st.button('Search'):
    books,images = recommend(selected_book) 
    #This image download step is improvised to bypass the problem of herokun not showing images through link.
    img1, _ = urllib.request.urlretrieve(images[0].strip())
    img2, _ = urllib.request.urlretrieve(images[1].strip())
    img3, _ = urllib.request.urlretrieve(images[2].strip())
    img4, _ = urllib.request.urlretrieve(images[3].strip())
    img5, _ = urllib.request.urlretrieve(images[4].strip())
    img6, _ = urllib.request.urlretrieve(images[5].strip())

    container1 =st.container()
    container1.subheader("You Searched For:")
    container1.markdown(books[0])
    container1.image(img1,width=120)

    st.subheader("Users Also Liked:")
    col1, col2, col3,col4,col5 = st.columns(5)

    with col1:
        st.text(books[1])
        st.image(img2,width=100)
    with col2:
        st.text(books[2])
        st.image(img3,width=100)
    with col3:
        st.text(books[3])
        st.image(img4,width=100)
    with col4:
        st.text(books[4])
        st.image(img5,width=100)
    with col5:
        st.text(books[5])
        st.image(img6,width=100)
