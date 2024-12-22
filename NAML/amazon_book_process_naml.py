import os
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pickle
from collections import Counter
import re
from nltk.tokenize import RegexpTokenizer
import urllib.request
import zipfile

def download_and_extract_glove(dest_path):
    """Download and extract GloVe embeddings.
    
    Args:
        dest_path (str): Destination directory path
        
    Returns:
        str: Path to extracted GloVe files
    """
    url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
    
    print(f"Downloading GloVe from {url}...")
    zip_path = os.path.join(dest_path, "glove.zip")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting GloVe files...")
    glove_path = os.path.join(dest_path, "glove")
    os.makedirs(glove_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(glove_path)
    
    os.remove(zip_path)
    return glove_path

def load_glove_matrix(path_emb, word_dict, word_embedding_dim):
    """Load pretrained GloVe embeddings for words in dictionary.
    
    Args:
        path_emb (str): Path to GloVe embeddings directory
        word_dict (dict): Word dictionary
        word_embedding_dim (int): Embedding dimension
        
    Returns:
        numpy.ndarray: Embedding matrix
        list: Words found in GloVe
        
    Raises:
        FileNotFoundError: If GloVe file not found
    """
    embedding_matrix = np.zeros((len(word_dict) + 1, word_embedding_dim))  # +1 for padding at index 0
    exist_word = []

    glove_file = os.path.join(path_emb, f"glove.6B.{word_embedding_dim}d.txt")
    print(f"Loading GloVe embeddings from {glove_file}...")
    
    try:
        with open(glove_file, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                if word in word_dict:
                    vector = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[word_dict[word]] = vector
                    exist_word.append(word)
    except FileNotFoundError:
        print(f"Error: GloVe file not found at {glove_file}")
        raise

    print(f"Found {len(exist_word)} words with embeddings out of {len(word_dict)} words")
    return embedding_matrix, exist_word

def word_tokenize(text):
    """Tokenize text thành các từ theo cách của MIND dataset.
    
    Args:
        text (str): Input text
        
    Returns:
        list: Danh sách các từ
    """
    # Xử lý các từ liên tiếp và dấu câu đặc biệt như MIND
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(text, str):
        return pat.findall(text.lower())
    return []

def truncate_text(text, max_length):
    """Cắt văn bản theo số từ tối đa.
    
    Args:
        text (str): Văn bản đầu vào
        max_length (int): Số từ tối đa
        
    Returns:
        list: Danh sách từ đã cắt
    """
    words = word_tokenize(text)
    return words[:max_length]

def get_books_in_behaviors(behaviors_file):
    """Lấy tất cả book IDs xuất hiện trong file behaviors.
    
    Args:
        behaviors_file (str): Đường dẫn đến file behaviors
        
    Returns:
        set: Tập hợp các book IDs
    """
    book_ids = set()
    with open(behaviors_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Lấy phần history và impressions
            history = line.strip().split('\t')[3]
            impressions = line.strip().split('\t')[4]
            
            # Thêm books từ history
            if history.strip():
                book_ids.update(history.split())
                
            # Thêm books từ impressions (bỏ phần label -1/-0)
            if impressions.strip():
                book_ids.update([imp.split('-')[0] for imp in impressions.split()])
    
    return book_ids

def generate_news_file(output_file, books_df, book_ids=None):
    """Generate news.tsv file in MIND format.
    Format: [News ID] [Category] [Subcategory] [News Title] [News Abstract] [News Url] [Entities in Title] [Entities in Abstract]
    
    Args:
        output_file (str): Đường dẫn file output
        books_df (DataFrame): DataFrame chứa thông tin sách
        book_ids (set, optional): Tập book IDs cần tạo news. Nếu None, tạo cho tất cả sách.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, book in books_df.iterrows():
            # Nếu có book_ids và book không nằm trong tập đó thì bỏ qua
            if book_ids is not None and book['item_id'] not in book_ids:
                continue
                
            # Convert title and description lists back to strings
            title_str = ' '.join(book['title']) if isinstance(book['title'], list) else ''
            desc_str = ' '.join(book['description']) if isinstance(book['description'], list) else ''
            
            line = [
                book['item_id'],                # News ID
                book['category'],               # Category (vertical)
                book['subcategory'],            # Subcategory (subvertical) 
                title_str,                      # Title
                desc_str,                       # Abstract/Description
                '',                             # URL (empty)
                '[]',                           # Title entities (empty)
                '[]'                            # Abstract entities (empty)
            ]
            f.write('\t'.join(line) + '\n')

def process_amazon_books(
    books_file,
    reviews_file,
    output_path,
    max_title_length=30,
    max_abstract_length=50,
    history_size=100,
    word_freq_threshold=1,
    word_embedding_dim=300,
    sample_size=None,
    train_neg_nums=4,  # Number of negative samples for training
    valid_neg_nums=20,  # Number of negative samples for validation
    test_neg_nums=20  # Number of negative samples for testing
):
   
    # Validate input files
    if not os.path.exists(books_file):
        raise FileNotFoundError(f"Books file not found: {books_file}")
    if not os.path.exists(reviews_file):
        raise FileNotFoundError(f"Reviews file not found: {reviews_file}")
        
    # Validate embedding dimension
    valid_dims = [50, 100, 200, 300]
    if word_embedding_dim not in valid_dims:
        raise ValueError(f"word_embedding_dim phải là một trong {valid_dims}")

    # Create directories
    os.makedirs(output_path, exist_ok=True)
    train_dir = os.path.join(output_path, 'train')
    valid_dir = os.path.join(output_path, 'valid')
    test_dir = os.path.join(output_path, 'test')
    utils_path = os.path.join(output_path, 'utils')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(utils_path, exist_ok=True)

    print("Loading and processing books metadata...")
    books_data = []
    num_books = sum(1 for _ in open(books_file, 'r', encoding="utf-8"))
    with open(books_file, 'r') as f:
        for line in tqdm(f, total=num_books):
            data = json.loads(line.strip())
            # Tokenize và cắt title và description theo độ dài tối đa
            title = truncate_text(data.get('title', ''), max_title_length)
            
            description_data = data.get('description', '')

            if isinstance(description_data, list):
                # Nối các chuỗi trong danh sách thành một chuỗi duy nhất, cách nhau bởi dấu cách
                description_data = ' '.join(description_data)

            # Cắt ngắn chuỗi đã nối (hoặc chuỗi gốc nếu không phải danh sách)
            description = truncate_text(description_data, max_abstract_length)
            
            subcategory = data.get('category', [])
            
            books_data.append({
                'item_id': data['asin'],
                'title': title,  # List of tokenized words
                'description': description,  # List of tokenized words
                'category': subcategory[0] if subcategory else "Unknown",
                'subcategory': subcategory[-1] if subcategory else "Unknown"
            })
    books_df = pd.DataFrame(books_data)
    print(f"Loaded {len(books_df)} books")
    
    print("\nLoading and processing reviews...")
    reviews_data = []
    num_reviews = sum(1 for _ in open(reviews_file, 'r'))
    with open(reviews_file, 'r') as f:
        for line in tqdm(f, total=num_reviews):
            data = json.loads(line.strip())
            reviews_data.append({
                'user_id': data['reviewerID'],
                'item_id': data['asin'],
                'timestamp': data['unixReviewTime'],
            })
    reviews_df = pd.DataFrame(reviews_data)
    print(f"Loaded {len(reviews_df)} reviews from {len(reviews_df['user_id'].unique())} users")
    
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
            
        # Tính số reviews tối thiểu cần thiết cho mỗi user
        min_reviews = history_size + 3  # Cho test split
        user_counts = reviews_df['user_id'].value_counts()
        qualified_users = user_counts[user_counts >= min_reviews]
        
        if len(qualified_users) == 0:
            raise ValueError(f"No users have enough reviews (minimum {min_reviews} reviews required)")
            
        sample_size = min(sample_size, len(qualified_users))
        user_sample = qualified_users.index[:sample_size]
        reviews_df = reviews_df[reviews_df['user_id'].isin(user_sample)]
        print(f"Sampled {sample_size} users with at least {min_reviews} reviews each")
    
   
    utils_files_exist = all(os.path.exists(os.path.join(utils_path, f)) for f in [
        'word_dict_all.pkl',
        'vert_dict.pkl', 
        'subvert_dict.pkl',
        'uid2index.pkl',

        f'embedding_all_{word_embedding_dim}d.npy'
    ])

    if utils_files_exist:
        print("\nUtils files already exist. Skipping dictionary and embedding creation...")
    else:
        # Create word dictionary
        print("\nCreating word dictionary...")
        word_counter = Counter()
        for title in tqdm(books_df['title'], desc="Processing titles"):
            word_counter.update(title)
        for desc in tqdm(books_df['description'], desc="Processing descriptions"):
            word_counter.update(desc)
            
        # Tạo word_dict theo cách của MIND
        filtered_words = [word for word, freq in word_counter.items() if freq >= word_freq_threshold]
        word_dict = {k: v+1 for k, v in zip(filtered_words, range(len(filtered_words)))}
        print(f"Created dictionary with {len(word_dict)} words (frequency >= {word_freq_threshold})")
        
        # Create category dictionaries
        category_dict = {cat: idx + 1 for idx, cat in 
                        enumerate(books_df['category'].unique())}
        subcategory_dict = {subcat: idx + 1 for idx, subcat in 
                           enumerate(books_df['subcategory'].unique())}
        
        # Create user dictionary
        user_dict = {user: idx + 1 for idx, user in 
                     enumerate(reviews_df['user_id'].unique())}
        
        # Save dictionaries
        print("\nSaving dictionaries...")
        with open(os.path.join(utils_path, 'word_dict_all.pkl'), 'wb') as f:
            pickle.dump(word_dict, f)
        with open(os.path.join(utils_path, 'vert_dict.pkl'), 'wb') as f:
            pickle.dump(category_dict, f)
        with open(os.path.join(utils_path, 'subvert_dict.pkl'), 'wb') as f:
            pickle.dump(subcategory_dict, f)
        with open(os.path.join(utils_path, 'uid2index.pkl'), 'wb') as f:
            pickle.dump(user_dict, f)

        # Download and process word embeddings if needed
        print("\nProcessing word embeddings...")
        glove_path = os.path.join(output_path, "glove")

        if not os.path.exists(glove_path):
            print(f"Path does not exist. Downloading glove embeddings to {glove_path}...")
            glove_path = download_and_extract_glove(output_path)
        else:
            print(f"Word embeddings already exist at {glove_path}")
        
        # Generate word embeddings
        embedding_matrix, exist_words = load_glove_matrix(
            glove_path, 
            word_dict,
            word_embedding_dim
        )
        
        # Save embeddings
        print("\nSaving word embeddings...")
        embedding_file = os.path.join(utils_path, f'embedding_all_{word_embedding_dim}d.npy')
        np.save(embedding_file, embedding_matrix)

    # Generate behaviors.tsv files for all splits in one pass
    print("\nGenerating behavior files for all splits...")
    behaviors_count = {'train': 0, 'valid': 0, 'test': 0}
    all_books = set()
    
    # Get all available book IDs for negative sampling
    all_book_ids = set(books_df['item_id'].unique())
    
    # Open all behavior files at once
    with open(os.path.join(train_dir, 'behaviors.tsv'), 'w', encoding='utf-8') as train_f, \
         open(os.path.join(valid_dir, 'behaviors.tsv'), 'w', encoding='utf-8') as valid_f, \
         open(os.path.join(test_dir, 'behaviors.tsv'), 'w', encoding='utf-8') as test_f:
        
        file_handlers = {'train': train_f, 'valid': valid_f, 'test': test_f}
        total_users = len(reviews_df['user_id'].unique())
        skipped_users = 0
        
        # Process each user
        for user_id in tqdm(reviews_df['user_id'].unique(), total=total_users):
            # Get and sort user's reviews by time
            user_reviews = reviews_df[reviews_df['user_id'] == user_id].sort_values('timestamp')
            total_reviews = len(user_reviews)
            
            if total_reviews < 3:  # Skip users with too few reviews
                skipped_users += 1
                continue
                
            # Process last 3 reviews for train, valid, test splits
            for i, split_name in enumerate(['train', 'valid', 'test']):
                review_idx = total_reviews - 3 + i
                
                # Get history items (limited by history_size)
                history_start = max(0, review_idx - history_size)
                history_items = user_reviews.iloc[history_start:review_idx]['item_id'].tolist()
                impression_item = user_reviews.iloc[review_idx]['item_id']
                
                # Add books to all_books set
                all_books.update(history_items)
                all_books.add(impression_item)
                
                # Create behavior line
                history_str = ' '.join(history_items)
                
                # Initialize impressions with positive sample
                impressions = [f"{impression_item}-1"]
                
                # Add negative samples for all splits
                # Get number of negative samples for this split
                neg_nums = {
                    'train': train_neg_nums,
                    'valid': valid_neg_nums,
                    'test': test_neg_nums
                }[split_name]
                
                # Get candidate negative items (excluding history and positive item)
                candidate_neg_items = all_book_ids - set(history_items) - {impression_item}
                
                # Sample negative items
                if len(candidate_neg_items) >= neg_nums:
                    neg_items = random.sample(candidate_neg_items, neg_nums)
                else:
                    # If not enough candidates, sample with replacement
                    neg_items = random.choices(list(candidate_neg_items), k=neg_nums)
                
                # Add negative samples to impressions
                impressions.extend([f"{item}-0" for item in neg_items])
                
                # Create final impression string
                impression_str = ' '.join(impressions)
                
                timestamp = datetime.fromtimestamp(
                    user_reviews.iloc[review_idx]['timestamp']
                ).strftime('%m/%d/%Y %H:%M:%S')
                
                line = [
                    str(behaviors_count[split_name]),  # Impression ID
                    user_id,                          # User ID
                    timestamp,                        # Time
                    history_str,                      # History
                    impression_str                    # Impressions
                ]
                
                # Write to appropriate file
                file_handlers[split_name].write('\t'.join(line) + '\n')
                behaviors_count[split_name] += 1
    
    print(f"Processed {total_users - skipped_users} users, skipped {skipped_users} users")
    output_dirs = {
       'train': train_dir,
       'valid': valid_dir,
       'test': test_dir
   }
    # Generate news.tsv files for all splits
    print("\nGenerating news files...")
    for split_name in ['train', 'valid', 'test']:
        output_file = os.path.join(output_dirs[split_name], 'news.tsv')
        generate_news_file(output_file, books_df, all_books)
    


    print("\nData processing completed!")
    stats = {
        'num_users': len(user_dict),
        'num_items': len(books_df),
        'num_categories': len(category_dict),
        'num_subcategories': len(subcategory_dict),
        'vocab_size': len(word_dict),
        'num_train_behaviors': behaviors_count['train'],
        'num_valid_behaviors': behaviors_count['valid'],
        'num_test_behaviors': behaviors_count['test'],
        'num_words_with_embeddings': len(exist_words)
    }
    
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    return stats

if __name__ == "__main__":
    # Example usage
    books_file =r"Web-Mining/datasets/meta_Books.json/meta_Books.json"
    reviews_file = r"Web-Mining/datasets/Books_5.json/Books_5.json"
    output_path = "processed_data"
    
    stats = process_amazon_books(
        books_file=books_file,
        reviews_file=reviews_file,
        output_path=output_path,
        word_embedding_dim=300,  # Use 300d GloVe embeddings
        sample_size=None  # Process all users instead of 5000
    )
    print("\nProcessing Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")