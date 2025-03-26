import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configuration
SAVE_PATH = "D:\\hand"
BASE_URL = "https://freepokertools.holdemmanager.com/hand/{}/convert/"
os.makedirs(SAVE_PATH, exist_ok=True)

def find_next_hand_id(start_id=37900000, finish_id=37937077, batch_size=100):
    """
    Generates hand IDs in batches between start_id and finish_id (inclusive)
    If finish_id is None, runs indefinitely
    """
    current_id = start_id
    while True:
        if finish_id is not None and current_id > finish_id:
            break
            
        upper = current_id + batch_size
        if finish_id is not None:
            upper = min(upper, finish_id + 1)  # +1 because range() is exclusive
        
        yield from range(current_id, upper)
        current_id = upper
        
        if finish_id is not None and current_id > finish_id:
            break

def setup_driver():
    return webdriver.Chrome()

def get_hand_history(driver, hand_id):
    url = BASE_URL.format(hand_id)
    try:
        driver.get(url)
        history_field = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "PT_history_field"))
        )
        return history_field.get_attribute("value")
    except Exception as e:
        print(f"Error fetching {hand_id}: {str(e)}")
        return None

def format_hand_history(hand_data):
    if not hand_data or "No Limit Hold'em Cash" not in hand_data:
        return None
    replacements = {
        "Poker Stars": "Poker Stars, $0.05/$0.10 No Limit Hold'em Cash, 6 Players",
        "♠": "s", "♣": "c", "♦": "d", "♥": "h",
        "Preflop": "Preflop: Hero is MP with Ks Qd",
        "Results": "Results: $20.89 pot ($0.94 rake)",
        "Final Board": "Final Board: 2d 4s Td Js 3c"
    }
    for old, new in replacements.items():
        hand_data = hand_data.replace(old, new)
    return hand_data

def save_hand_history(hand_data, hand_id):
    if not hand_data:
        return False
    file_path = os.path.join(SAVE_PATH, f"hand_{hand_id}.txt")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(hand_data)
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False

def scrape_hands(start_id=37900000, finish_id=37937077):
    stats = {'checked': 0, 'saved': 0, 'errors': 0, 'reasons': {}}
    driver = setup_driver()

    try:
        for hand_id in find_next_hand_id(start_id=start_id, finish_id=finish_id):
            try:
                hand_data = get_hand_history(driver, hand_id)
                stats['checked'] += 1

                if hand_data:
                    formatted = format_hand_history(hand_data)
                    if formatted:
                        if save_hand_history(formatted, hand_id):
                            stats['saved'] += 1
                            print(f"✅ Saved {hand_id}")
                        else:
                            stats['errors'] += 1
                            stats['reasons']['Save error'] = stats['reasons'].get('Save error', 0) + 1
                    else:
                        print(f"❌ Skipped {hand_id} (Wrong game type)")
                else:
                    stats['errors'] += 1
                    stats['reasons']['Fetch error'] = stats['reasons'].get('Fetch error', 0) + 1

                if stats['checked'] % 100 == 0:
                    print(f"\nProgress: Checked {stats['checked']}, Saved {stats['saved']}, Errors {stats['errors']}")

                time.sleep(0.5)

            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                break
            except Exception as e:
                print(f"Critical error: {str(e)}")
                continue

    finally:
        driver.quit()
        print("\nFinal Statistics:")
        print(f"Total checked: {stats['checked']}")
        print(f"Successfully saved: {stats['saved']}")
        print(f"Errors encountered: {stats['errors']}")
        if stats['reasons']:
            print("Error reasons:")
            for reason, count in stats['reasons'].items():
                print(f" - {reason}: {count}")

if __name__ == "__main__":
    # Example usage:
    scrape_hands(start_id=37900000, finish_id=37937077)  # Specific range
    #scrape_hands()  # Run indefinitely from default start ID