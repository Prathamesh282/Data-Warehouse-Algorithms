import pandas as pd
from itertools import combinations

# Sample transactional data
data = [
    ['Milk', 'Bread', 'Diaper'],
    ['Milk', 'Bread'],
    ['Bread', 'Diaper'],
    ['Milk', 'Diaper'],
    ['Bread', 'Diaper'],
    ['Milk']
]

# Function to generate frequent itemsets
def get_frequent_itemsets(data, min_support):
    item_count = {}  # Initialize item_count dictionary
    total_transactions = len(data)
    
    # Count individual items
    for transaction in data:
        for item in transaction:
            item_count[item] = item_count.get(item, 0) + 1

    # Filter by minimum support
    frequent_itemsets = {frozenset([item]): count for item, count in item_count.items() if count / total_transactions >= min_support}

    k = 2  # Start with 2-item combinations
    while True:
        itemsets = combinations(frequent_itemsets.keys(), k)
        itemset_count = {}  # Initialize for k-itemsets
        
        # Count itemset occurrences
        for transaction in data:
            transaction_set = set(transaction)
            for itemset in itemsets:
                if set(itemset).issubset(transaction_set):
                    itemset_count[itemset] = itemset_count.get(itemset, 0) + 1
        
        # Filter by support
        new_frequent_itemsets = {frozenset(itemset): count for itemset, count in itemset_count.items() if count / total_transactions >= min_support}

        if not new_frequent_itemsets:
            break
        
        frequent_itemsets.update(new_frequent_itemsets)
        k += 1

    return frequent_itemsets

# Function to generate association rules
def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            for antecedent in combinations(itemset, len(itemset) - 1):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                if consequent:
                    confidence = frequent_itemsets[itemset] / frequent_itemsets[antecedent]
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules

# Parameters
min_support = 0.3
min_confidence = 0.7

# Execution
frequent_itemsets = get_frequent_itemsets(data, min_support)
rules = generate_association_rules(frequent_itemsets, min_confidence)

# Display results
print("Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"{set(itemset)}: {count}")

print("\nAssociation Rules:")
for antecedent, consequent, confidence in rules:
    print(f"{set(antecedent)} -> {set(consequent)} (Confidence: {confidence:.2f})")