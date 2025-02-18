import cv2

def match_images(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Use BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance = better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate matching percentage
    match_percentage = (len(matches) / max(len(kp1), len(kp2))) * 100

    # Draw matches
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display result
    cv2.imshow('Matches', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return match_percentage

# Example usage
img1_path = "C:\\Users\\Admin\\Pictures\\New folder (2)\\New folder (4)\\1737984588412.jpg"  # Replace with your image paths
img2_path = "C:\\Users\\Admin\\Pictures\\New folder (2)\\New folder (4)\\1737984588448.jpg"
percentage = match_images(img1_path, img2_path)
print(f"Matching Percentage: {percentage:.2f}%")