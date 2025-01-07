def chunk_array(data):
    """
    Splits an array into chunks of a specified size.
    
    Args:
        data (list): The list to be split.
        chunk_size (int): The size of each chunk.
    
    Returns:
        list: A list containing sublists (chunks).
    """
    for i in range(0, len(data), 600):
        yield data[i:i + 600]
    
    return data

#def process_chunk(chunk):
    """
    Example function to process each chunk.
    
    Args:
        chunk (list): A list containing the chunk of data.
    """
#    print(f"Processing chunk of size {len(chunk)}")
    # Your processing logic here

# Example data
#data = [{'id': i, 'value': f'value_{i}'} for i in range(1, 2500)]

# Chunk size
#chunk_size = 600

# Split data and process each chunk
#for chunk in chunk_array(data, chunk_size):
#    process_chunk(chunk)
