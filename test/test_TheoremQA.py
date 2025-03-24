import sys
import os

# Add the parent directory to the path so we can import the dynamic_penalty module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamic_penalty.data.TheoremQA import get_theoremqa_questions, extract_xml_answer

def test_theoremqa_dataset():
    """Test loading and processing the TheoremQA dataset."""
    print("Testing TheoremQA dataset loading...")
    
    # Print the total size of the dataset
    full_dataset = get_theoremqa_questions(split="test")
    print(f"Total TheoremQA dataset size: {len(full_dataset)} examples")
    
    # Test with a small limit to make it faster
    dataset = get_theoremqa_questions(split="test", limit=5)
    
    print(f"Successfully loaded dataset with {len(dataset)} examples")
    
    # Print the first example to verify structure
    example = dataset[0]
    print("\nExample prompt:")
    for message in example['prompt']:
        print(f"Role: {message['role']}")
        print(f"Content: {message['content']}")
    
    print(f"\nExample answer: {example['answer']}")
    print(f"Answer type: {example['answer_type']}")
    
    # Print dataset fields
    print("\nDataset fields:")
    print(list(example.keys()))
    
    # Count examples by answer type
    answer_types = {}
    for item in full_dataset:
        answer_type = item['answer_type']
        if answer_type in answer_types:
            answer_types[answer_type] += 1
        else:
            answer_types[answer_type] = 1
    
    print("\nDistribution of answer types:")
    for answer_type, count in answer_types.items():
        print(f"  {answer_type}: {count} examples ({count/len(full_dataset)*100:.1f}%)")
    
    # Test the extract_xml_answer function
    print("\nTesting XML extraction...")
    test_xml = "<reasoning>This is a test reasoning.</reasoning><answer>Test answer</answer>"
    extracted = extract_xml_answer(test_xml)
    print(f"Test XML: {test_xml}")
    print(f"Extracted answer from XML: '{extracted}'")
    assert extracted == "Test answer", "XML extraction failed"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_theoremqa_dataset() 