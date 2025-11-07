# import pandas as pd
# import numpy as np
# import joblib
# from sentence_transformers import SentenceTransformer
# import networkx as nx
# from flask import Flask, render_template, request
# import re
# import unicodedata
# from text_unidecode import unidecode
# import ast # To safely evaluate string lists

# # Use the correct import for Levenshtein functions
# import Levenshtein
# from Levenshtein import jaro_winkler

# from sklearn.metrics.pairwise import cosine_similarity

# # =============================================
# # 1. Initialize Flask App and Load Models
# # =============================================
# print("Initializing Flask app and loading models...")
# app = Flask(__name__)

# # Load all our pre-trained components
# try:
#     model = joblib.load('and_model.pkl')
#     sbert_model = SentenceTransformer('sbert_model')
#     df = pd.read_csv('publication_database.csv').set_index('publication_id')

#     # --- CORE FIX: Ensure all titles are strings BEFORE encoding ---
#     # This prevents errors from NaN or numeric values in the title column.
#     df['title'] = df['title'].astype(str)
    
#     # Pre-compute SBERT embeddings for the entire database (one-time cost)
#     print("Pre-computing SBERT embeddings for the database. This may take a moment...")
#     sbert_embeddings = sbert_model.encode(df['title'].tolist(), show_progress_bar=True)
#     sbert_map = dict(zip(df.index, sbert_embeddings))
    
#     print("Models and data loaded successfully.")
# except FileNotFoundError as e:
#     print(f"FATAL ERROR: Could not find a required file. {e}")
#     print("Please make sure 'and_model.pkl', 'sbert_model/', and 'publication_database.csv' are in the same directory.")
#     exit()

# # =============================================
# # 2. Helper Functions
# # =============================================
# def normalize_name_query(name):
#     if not isinstance(name, str) or not name: return ""
#     name = unidecode(name).lower()
#     name = re.sub(r'[^a-z\s,]', '', name)
#     parts = [p.strip() for p in re.split(r'[\s,]+', name) if p.strip()]
#     if not parts: return ""
#     lastname = parts[-1]; initials = [p[0] for p in parts[:-1]]
#     return " ".join(initials) + " " + lastname

# def compute_features_live(pub1, pub2, sbert_map_subset):
#     """Computes features for a pair of publications."""
#     features = {}
#     features['name_jaro'] = jaro_winkler(pub1['norm_author_name'], pub2['norm_author_name'])
#     aff1 = set(str(pub1['norm_affiliation']).split()); aff2 = set(str(pub2['norm_affiliation']).split())
#     features['aff_jaccard'] = len(aff1.intersection(aff2)) / len(aff1.union(aff2)) if aff1.union(aff2) else 0
#     coauth1 = set(ast.literal_eval(pub1['norm_co_authors'])) if isinstance(pub1['norm_co_authors'], str) and pub1['norm_co_authors'].startswith('[') else set()
#     coauth2 = set(ast.literal_eval(pub2['norm_co_authors'])) if isinstance(pub2['norm_co_authors'], str) and pub2['norm_co_authors'].startswith('[') else set()
#     features['coauth_jaccard'] = len(coauth1.intersection(coauth2)) / len(coauth1.union(coauth2)) if coauth1.union(coauth2) else 0
#     venue1, venue2 = str(pub1['norm_venue']), str(pub2['norm_venue']); max_len = max(len(venue1), len(venue2))
#     features['venue_lev'] = 1 - (Levenshtein.distance(venue1, venue2) / max_len) if max_len > 0 else 1
#     features['year_prox'] = np.exp(-0.1 * abs(pub1['year'] - pub2['year']))
#     emb1 = sbert_map_subset[pub1.name]; emb2 = sbert_map_subset[pub2.name]
#     features['title_sbert_sim'] = cosine_similarity([emb1], [emb2])[0][0]
#     return features

# # =============================================
# # 3. Flask Routes
# # =============================================
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         author_name_query = request.form['author_name']
#         norm_query = normalize_name_query(author_name_query)
#         surname = norm_query.split()[-1] if norm_query else ''
#         potential_matches = df[df['norm_author_name'].str.contains(surname, na=False)]
        
#         if potential_matches.empty:
#             return render_template('index.html', author_name=author_name_query, results={}, cluster_count=0, total_publications=0)
            
#         pub_ids = potential_matches.index.tolist()
#         candidate_pairs = []
#         for i in range(len(pub_ids)):
#             for j in range(i + 1, len(pub_ids)):
#                 candidate_pairs.append((pub_ids[i], pub_ids[j]))
                
#         if not candidate_pairs:
#              clusters = [[pub_ids[0]]]
#         else:
#             X_live = []
#             sbert_map_subset = {pid: sbert_map[pid] for pid in pub_ids}
#             for id1, id2 in candidate_pairs:
#                 pub1 = potential_matches.loc[id1]; pub2 = potential_matches.loc[id2]
#                 features = compute_features_live(pub1, pub2, sbert_map_subset)
#                 X_live.append(list(features.values()))

#             pair_probabilities = model.predict_proba(np.array(X_live))[:, 1]
#             G = nx.Graph()
#             G.add_nodes_from(pub_ids)
#             for i, (id1, id2) in enumerate(candidate_pairs):
#                 if pair_probabilities[i] > 0.5:
#                     G.add_edge(id1, id2, weight=pair_probabilities[i])
#             clusters = list(nx.connected_components(G))

#         results_dict = {}
#         for i, cluster in enumerate(clusters):
#             results_dict[i] = potential_matches.loc[list(cluster)].to_dict('records')
            
#         return render_template('index.html', 
#                                author_name=author_name_query, 
#                                results=results_dict, 
#                                cluster_count=len(clusters), 
#                                total_publications=len(potential_matches))

#     return render_template('index.html', author_name=None, results={})

# # =============================================
# # 4. Run the App
# # =============================================
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

#-----------------------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import joblib
# from sentence_transformers import SentenceTransformer
# import networkx as nx
# from flask import Flask, render_template, request
# import re
# import unicodedata
# from text_unidecode import unidecode
# import ast

# import Levenshtein
# from Levenshtein import jaro_winkler

# from sklearn.metrics.pairwise import cosine_similarity

# # =============================================
# # 1. DEFINE HELPER FUNCTIONS FIRST
# # =============================================
# # This function must be defined before it is used during data loading.
# def normalize_text_generic(text):
#     if not isinstance(text, str): return ""
#     text = unidecode(text).lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     return re.sub(r'\s+', ' ', text).strip()

# def compute_features_live(pub1, pub2, sbert_map_subset):
#     features = {}
#     features['name_jaro'] = jaro_winkler(pub1['norm_author_name'], pub2['norm_author_name'])
#     features['aff_jaccard'] = len(set(pub1['norm_affiliation'].split()) & set(pub2['norm_affiliation'].split())) / len(set(pub1['norm_affiliation'].split()) | set(pub2['norm_affiliation'].split())) if (set(pub1['norm_affiliation'].split()) | set(pub2['norm_affiliation'].split())) else 0
#     coauth1 = set(ast.literal_eval(pub1['norm_co_authors'])) if pub1['norm_co_authors'].startswith('[') else set()
#     coauth2 = set(ast.literal_eval(pub2['norm_co_authors'])) if pub2['norm_co_authors'].startswith('[') else set()
#     features['coauth_jaccard'] = len(coauth1 & coauth2) / len(coauth1 | coauth2) if (coauth1 | coauth2) else 0
#     venue1, venue2 = pub1['norm_venue'], pub2['norm_venue']; max_len = max(len(venue1), len(venue2))
#     features['venue_lev'] = 1 - (Levenshtein.distance(venue1, venue2) / max_len) if max_len > 0 else 1
#     features['year_prox'] = np.exp(-0.1 * abs(pub1['year'] - pub2['year']))
#     emb1, emb2 = sbert_map_subset[pub1.name], sbert_map_subset[pub2.name]
#     features['title_sbert_sim'] = cosine_similarity([emb1], [emb2])[0][0]
#     return features


# # =============================================
# # 2. Initialize Flask App and Load Models
# # =============================================
# print("Initializing Flask app and loading models...")
# app = Flask(__name__)

# try:
#     model = joblib.load('and_model.pkl')
#     sbert_model = SentenceTransformer('sbert_model')
    
#     # Load the database using the correct primary key
#     df = pd.read_csv('publication_database.csv').set_index('publication_id')

#     # Data cleaning and type correction
#     if 'publication_id.1' in df.columns:
#         df = df.drop(columns=['publication_id.1'])
        
#     text_cols_to_clean = [
#         'title', 'norm_affiliation', 'Country', 'norm_venue', 'norm_author_name',
#         'author_name', 'affiliation', 'venue'
#     ]
#     for col in text_cols_to_clean:
#         if col in df.columns:
#             df[col] = df[col].astype(str).fillna('')
            
#     # Add a normalized country column for the filter to use
#     if 'Country' in df.columns and 'norm_country' not in df.columns:
#         # NOW this call will work because the function is defined above
#         df['norm_country'] = df['Country'].apply(lambda x: normalize_text_generic(x))

#     print("Pre-computing SBERT embeddings for all titles...")
#     sbert_embeddings = sbert_model.encode(df['title'].tolist(), show_progress_bar=True)
#     sbert_map = dict(zip(df.index, sbert_embeddings))
#     print("Models and data loaded successfully.")
# except FileNotFoundError as e:
#     print(f"FATAL ERROR: A required file was not found. {e}")
#     exit()
# except Exception as e:
#     print(f"An error occurred during model loading: {e}")
#     exit()


# # =============================================
# # 3. Flask Route with Advanced Filtering
# # =============================================
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         search_criteria = {k: v.strip() for k, v in request.form.items()}
#         norm_last_name = normalize_text_generic(search_criteria.get('last_name', ''))
#         norm_first_name = normalize_text_generic(search_criteria.get('first_name', ''))
#         norm_affiliation = normalize_text_generic(search_criteria.get('affiliation', ''))
#         norm_country = normalize_text_generic(search_criteria.get('country', ''))

#         if not norm_last_name:
#             return render_template('index.html', search_criteria=search_criteria, results={}, error="Last Name is required.")

#         mask = df['norm_author_name'].str.contains(f"\\b{norm_last_name}\\b", regex=True)
#         if norm_first_name:
#             first_initial = norm_first_name.split()[0][0] if norm_first_name.split() else ''
#             if first_initial:
#                 mask &= df['norm_author_name'].str.startswith(first_initial)
#         if norm_affiliation:
#             mask &= df['norm_affiliation'].str.contains(norm_affiliation)
#         if norm_country and 'norm_country' in df.columns:
#             mask &= df['norm_country'].str.contains(norm_country)
        
#         potential_matches = df[mask]

#         if potential_matches.empty:
#             return render_template('index.html', search_criteria=search_criteria, results={}, cluster_count=0, total_publications=0)

#         pub_ids = potential_matches.index.tolist()
#         clusters = [[pid] for pid in pub_ids]
#         if len(pub_ids) > 1:
#             candidate_pairs = [(pub_ids[i], pub_ids[j]) for i in range(len(pub_ids)) for j in range(i + 1, len(pub_ids))]
#             sbert_map_subset = {pid: sbert_map[pid] for pid in pub_ids}
#             X_live = [list(compute_features_live(potential_matches.loc[id1], potential_matches.loc[id2], sbert_map_subset).values()) for id1, id2 in candidate_pairs]
#             pair_probabilities = model.predict_proba(np.array(X_live))[:, 1]
#             G = nx.Graph()
#             G.add_nodes_from(pub_ids)
#             for i, (id1, id2) in enumerate(candidate_pairs):
#                 if pair_probabilities[i] > 0.5:
#                     G.add_edge(id1, id2)
            
#             connected_comps = list(nx.connected_components(G))
#             clustered_nodes = {node for comp in connected_comps for node in comp}
#             singletons = [[node] for node in pub_ids if node not in clustered_nodes]
#             clusters = connected_comps + singletons

#         results_dict = {}
#         for i, cluster_nodes in enumerate(clusters):
#             display_data = potential_matches.loc[list(cluster_nodes), ['title', 'year', 'author_name', 'venue', 'affiliation']]
#             results_dict[i] = display_data.to_dict('records')
            
#         return render_template('index.html', 
#                                search_criteria=search_criteria, 
#                                results=results_dict, 
#                                cluster_count=len(clusters), 
#                                total_publications=len(potential_matches))

#     # GET request: Show empty form
#     return render_template('index.html', search_criteria={}, results={})

# # =============================================
# # 4. Run the App
# # =============================================
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)



import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import networkx as nx
from flask import Flask, render_template, request
import re
import unicodedata
from text_unidecode import unidecode
import ast

import Levenshtein
from Levenshtein import jaro_winkler

from sklearn.metrics.pairwise import cosine_similarity

# =============================================
# 1. DEFINE HELPER FUNCTIONS FIRST
# =============================================
def normalize_text_generic(text):
    if not isinstance(text, str): return ""
    text = unidecode(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def compute_features_live(pub1, pub2, sbert_map_subset):
    features = {}
    features['name_jaro'] = jaro_winkler(pub1['norm_author_name'], pub2['norm_author_name'])
    features['aff_jaccard'] = len(set(pub1['norm_affiliation'].split()) & set(pub2['norm_affiliation'].split())) / len(set(pub1['norm_affiliation'].split()) | set(pub2['norm_affiliation'].split())) if (set(pub1['norm_affiliation'].split()) | set(pub2['norm_affiliation'].split())) else 0
    coauth1 = set(ast.literal_eval(pub1['norm_co_authors'])) if pub1['norm_co_authors'].startswith('[') else set()
    coauth2 = set(ast.literal_eval(pub2['norm_co_authors'])) if pub2['norm_co_authors'].startswith('[') else set()
    features['coauth_jaccard'] = len(coauth1 & coauth2) / len(coauth1 | coauth2) if (coauth1 | coauth2) else 0
    venue1, venue2 = pub1['norm_venue'], pub2['norm_venue']; max_len = max(len(venue1), len(venue2))
    features['venue_lev'] = 1 - (Levenshtein.distance(venue1, venue2) / max_len) if max_len > 0 else 1
    features['year_prox'] = np.exp(-0.1 * abs(pub1['year'] - pub2['year']))
    emb1, emb2 = sbert_map_subset[pub1.name], sbert_map_subset[pub2.name]
    features['title_sbert_sim'] = cosine_similarity([emb1], [emb2])[0][0]
    return features

# =============================================
# 2. Initialize Flask App and Load Models
# =============================================
print("Initializing Flask app and loading models...")
app = Flask(__name__)

try:
    model = joblib.load('and_model.pkl')
    sbert_model = SentenceTransformer('sbert_model')
    df = pd.read_csv('publication_database.csv').set_index('publication_id')

    # Data cleaning and type correction
    if 'publication_id.1' in df.columns:
        df = df.drop(columns=['publication_id.1'])
    
    text_cols_to_clean = [
        'title', 'norm_affiliation', 'Country', 'norm_venue', 'norm_author_name',
        'author_name', 'affiliation', 'venue', 'Emailid', 'RID', 'OID', 'Keyword'
    ]
    for col in text_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
    
    # Create normalized columns for searching
    if 'Country' in df.columns: df['norm_country'] = df['Country'].apply(normalize_text_generic)
    if 'Keyword' in df.columns: df['norm_keyword'] = df['Keyword'].apply(normalize_text_generic)

    print("Pre-computing SBERT embeddings...")
    sbert_embeddings = sbert_model.encode(df['title'].tolist(), show_progress_bar=True)
    sbert_map = dict(zip(df.index, sbert_embeddings))
    print("Models and data loaded successfully.")
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    exit()

# =============================================
# 3. Flask Route with Advanced Filtering
# =============================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_criteria = {k: v.strip() for k, v in request.form.items()}
        
        # Normalize all user inputs
        norm_last_name = normalize_text_generic(search_criteria.get('last_name', ''))
        norm_first_name = normalize_text_generic(search_criteria.get('first_name', ''))
        norm_affiliation = normalize_text_generic(search_criteria.get('affiliation', ''))
        norm_email = search_criteria.get('email', '') # Email is case-sensitive, don't normalize fully
        norm_keyword = normalize_text_generic(search_criteria.get('keyword', ''))
        norm_oid = search_criteria.get('oid', '')
        norm_rid = search_criteria.get('rid', '')

        if not norm_last_name:
            return render_template('index.html', search_criteria=search_criteria, results={}, error="Last Name is required.")

        # Multi-stage filtering
        mask = df['norm_author_name'].str.contains(f"\\b{norm_last_name}\\b", regex=True)
        if norm_first_name:
            initial = norm_first_name.split()[0][0] if norm_first_name.split() else ''
            if initial: mask &= df['norm_author_name'].str.startswith(initial)
        if norm_affiliation: mask &= df['norm_affiliation'].str.contains(norm_affiliation)
        if norm_email: mask &= df['Emailid'].str.contains(norm_email, na=False)
        if norm_keyword: mask &= df['norm_keyword'].str.contains(norm_keyword, na=False)
        if norm_oid: mask &= df['OID'].str.contains(norm_oid, na=False)
        if norm_rid: mask &= df['RID'].str.contains(norm_rid, na=False)
        
        potential_matches = df[mask]

        if potential_matches.empty:
            return render_template('index.html', search_criteria=search_criteria, results={}, cluster_count=0, total_publications=0)

        pub_ids = potential_matches.index.tolist()
        clusters = [[pid] for pid in pub_ids]
        if len(pub_ids) > 1:
            candidate_pairs = [(pub_ids[i], pub_ids[j]) for i in range(len(pub_ids)) for j in range(i + 1, len(pub_ids))]
            sbert_map_subset = {pid: sbert_map[pid] for pid in pub_ids}
            X_live = [list(compute_features_live(potential_matches.loc[id1], potential_matches.loc[id2], sbert_map_subset).values()) for id1, id2 in candidate_pairs]
            pair_probabilities = model.predict_proba(np.array(X_live))[:, 1]
            G = nx.Graph()
            G.add_nodes_from(pub_ids)
            for i, (id1, id2) in enumerate(candidate_pairs):
                if pair_probabilities[i] > 0.5:
                    G.add_edge(id1, id2)
            
            connected_comps = list(nx.connected_components(G))
            clustered_nodes = {node for comp in connected_comps for node in comp}
            singletons = [[node] for node in pub_ids if node not in clustered_nodes]
            clusters = connected_comps + singletons

        # Format results for display, including the new columns
        results_dict = {}
        for i, cluster_nodes in enumerate(clusters):
            display_cols = ['title', 'year', 'author_name', 'venue', 'affiliation', 'Emailid', 'RID', 'OID', 'Keyword']
            display_data = potential_matches.loc[list(cluster_nodes), display_cols]
            results_dict[i] = display_data.to_dict('records')
            
        return render_template('index.html', 
                               search_criteria=search_criteria, 
                               results=results_dict, 
                               cluster_count=len(clusters), 
                               total_publications=len(potential_matches))

    # GET request: Show empty form
    return render_template('index.html', search_criteria={}, results={})

# =============================================
# 4. Run the App
# =============================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)