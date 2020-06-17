from scipy.spatial import distance as scipy_distance

def print_content_summary(centers, coordinates, labels, tweets):
    for i in range(len(centers)):
        print("\n======================")
        print('Cluster {} ({} tweets) is approximated by:'.format(i, list(labels).count(i)))

        distances = scipy_distance.cdist([centers[i]], coordinates, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:5]:
            print("- ", tweets['tweet_text_clean'].values[idx].strip(), "(Score: %.3f)" % (1-distance))