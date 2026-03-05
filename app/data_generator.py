# data_generator.py
import json
import random
import argparse
from typing import List, Dict

def generate_training_data(buildings: List[str]) -> List[Dict]:
    """
    Generate synthetic training data for NLP models
    """
    
    # Intent patterns with variations
    patterns = {
        'navigation': [
            "Where is {building}?",
            "How do I get to {building}?",
            "Directions to {building} please",
            "Take me to {building}",
            "I need to find {building}",
            "Location of {building}",
            "Show me the way to {building}",
            "Navigate to {building}",
            # Swahili
            "Iko wapi {building}?",
            "Naenda wapi {building}?",
            "Nifikie wapi {building}?",
            "Elekea {building}",
            "Naomba maelekezo ya {building}"
        ],
        'information': [
            "What is {building}?",
            "Tell me about {building}",
            "Information about {building}",
            "Describe {building}",
            "What facilities are in {building}?",
            # Swahili
            "Ni nini {building}?",
            "Elezea kuhusu {building}",
            "Habari za {building}"
        ],
        'hours': [
            "When does {building} open?",
            "What are the opening hours of {building}?",
            "Closing time for {building}",
            "Is {building} open now?",
            # Swahili
            "{building} inafungua lini?",
            "Muda wa kufungulia {building}"
        ],
        'contact': [
            "Contact information for {building}",
            "Phone number of {building}",
            "Email of {building}",
            "How to contact {building}",
            # Swahili
            "Mawasiliano ya {building}",
            "Nambari ya simu ya {building}"
        ]
    }
    
    training_data = []
    
    for building in buildings:
        for intent, intent_patterns in patterns.items():
            for pattern in intent_patterns:
                # Generate variations
                query = pattern.format(building=building)
                
                # Add to training data
                training_data.append({
                    'query': query,
                    'intent': intent,
                    'building': building,
                    'language': 'swahili' if any(word in query.lower() 
                                               for word in ['iko', 'wapi', 'naenda', 'elekea', 'ni nini', 'elezea', 'habari']) 
                                else 'english'
                })
                
                # Add variations with typos (10% of queries)
                if random.random() < 0.1:
                    typo_query = add_typos(query)
                    training_data.append({
                        'query': typo_query,
                        'intent': intent,
                        'building': building,
                        'language': 'swahili' if any(word in query.lower() 
                                                   for word in ['iko', 'wapi', 'naenda', 'elekea', 'ni nini', 'elezea', 'habari']) 
                                    else 'english'
                    })
                
                # Add casual variations (20% of queries)
                if random.random() < 0.2:
                    casual_query = add_casual_variation(query)
                    training_data.append({
                        'query': casual_query,
                        'intent': intent,
                        'building': building,
                        'language': 'swahili' if any(word in query.lower() 
                                                   for word in ['iko', 'wapi', 'naenda', 'elekea', 'ni nini', 'elezea', 'habari']) 
                                    else 'english'
                    })
    
    return training_data

def add_typos(text: str) -> str:
    """Add realistic typos to text"""
    typos = {
        'where': 'were', 'the': 'teh', 'library': 'libary',
        'administration': 'adminstration', 'cafeteria': 'cafetaria',
        'university': 'universty', 'building': 'builing',
        'directions': 'diretions', 'please': 'plese',
        'information': 'infromation', 'about': 'abut',
        'iko': 'iko', 'wapi': 'wap', 'naenda': 'nenda',
        'elekea': 'eleka', 'maelekezo': 'maelezo'
    }
    
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in typos and random.random() < 0.3:
            words[i] = typos[word.lower()]
    
    return ' '.join(words)

def add_casual_variation(text: str) -> str:
    """Add casual/slang variations"""
    variations = {
        'Where is': ['Wheres', 'Where\'s', 'Where can I find'],
        'How do I get to': ['How to get to', 'How can I reach'],
        'Directions to': ['Way to', 'Path to'],
        'Take me to': ['Bring me to', 'Guide me to'],
        'What is': ['What\'s', 'Tell me what is'],
        'Tell me about': ['Tell me more about', 'What can you tell me about']
    }
    
    for formal, casual_list in variations.items():
        if formal in text:
            casual = random.choice(casual_list)
            text = text.replace(formal, casual)
            break
    
    # Add casual endings
    casual_endings = ['', ' please', ' thanks', ' thank you', ' pls', ' thx']
    if not any(ending in text for ending in casual_endings[1:]):
        text += random.choice(casual_endings)
    
    return text

def main():
    parser = argparse.ArgumentParser(description='Generate training data for AI models')
    parser.add_argument('--buildings', type=str, required=True, 
                       help='Text file containing building names (one per line)')
    parser.add_argument('--output', type=str, default='training_data.json',
                       help='Output JSON file (default: training_data.json)')
    
    args = parser.parse_args()
    
    # Read building names from file
    try:
        with open(args.buildings, 'r', encoding='utf-8') as f:
            buildings = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Building file '{args.buildings}' not found.")
        return
    
    print(f"📚 Found {len(buildings)} buildings")
    print("🔄 Generating training data...")
    
    # Generate training data
    training_data = generate_training_data(buildings)
    
    # Save to JSON file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {len(training_data)} training examples")
    print(f"💾 Saved to: {args.output}")
    
    # Show sample data
    print("\n📝 Sample training data:")
    for i, item in enumerate(training_data[:5]):
        print(f"{i+1}. Query: {item['query']}")
        print(f"   Intent: {item['intent']}, Building: {item['building']}")
        print(f"   Language: {item['language']}")
        print()

if __name__ == '__main__':
    main()