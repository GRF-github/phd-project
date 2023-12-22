import bz2
import pickle


with bz2.BZ2File('./rt_data/taas_data/fingerprints.pklz', 'rb') as f:
    fingerprints = pickle.load(f)

print(type(fingerprints))





# Save the dictionary with all the data in a file called 'related.pklz'
print("Saving the classifications' dictionary")
with bz2.BZ2File('./rt_data/fingerprints.pklz', 'wb') as f:
    pickle.dump(fingerprints, f)
