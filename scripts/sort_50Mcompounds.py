import zipfile
import os
import shutil


# Open the zip file and extract its contents
if os.path.exists('/home/grf/PycharmProjects/cmmrt/resources/classifications_50Mcompounds.zip'):
    print('Extracting')
    with zipfile.ZipFile('/home/grf/PycharmProjects/cmmrt/resources/classifications_50Mcompounds.zip', 'r') as zip_ref:
        zip_ref.extractall('/home/grf/PycharmProjects/cmmrt/resources/')
    os.remove('/home/grf/PycharmProjects/cmmrt/resources/classifications_50Mcompounds.zip')

if os.path.exists('/home/grf/PycharmProjects/cmmrt/resources/all_classified.tsv'):
    print("Hashing")
    # Create Classification_files folder
    os.makedirs('/home/grf/PycharmProjects/cmmrt/resources/Classification_files/')

    # Open the file and read the first line
    with open('/home/grf/PycharmProjects/cmmrt/resources/all_classified.tsv', 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip any leading/trailing whitespace characters
            line = line.strip()  # Probably this line is not necessary
            # Split the line at the first tab character and get the first part
            InChI_key = line.split('\t')[0]  # Probably the '\t' inside the parenthesis is not necessary

            with open(f'/home/grf/PycharmProjects/cmmrt/resources/Classification_files/classification_file_{InChI_key[:3]}', 'a') as f:
                f.write(line + '\n')  # Probably adding '\n' does nothing

    os.remove('/home/grf/PycharmProjects/cmmrt/resources/all_classified.tsv')

if os.path.exists('/home/grf/PycharmProjects/cmmrt/resources/Classification_files/'):
    print("Sorting")

    # Get a list of all files in the directory
    list_of_files = os.listdir('/home/grf/PycharmProjects/cmmrt/resources/Classification_files/')
    # Sort the list of files in alphabetical order
    list_of_files.sort()

    for file in list_of_files:
        # Read all lines from the first file into a list
        with open(f'/home/grf/PycharmProjects/cmmrt/resources/Classification_files/{file}', 'r') as file_to_read:
            list_of_lines = file_to_read.readlines()

        # Order the lines based on the alphabetical order of the first string of each line
        list_of_lines.sort(key=lambda x: x.split()[0])

        # Save the ordered lines back to the same .tsv file, overwriting it
        with open('/home/grf/PycharmProjects/cmmrt/resources/all_classified_sorted.tsv', 'a') as file_to_write:
            for line in list_of_lines:
                file_to_write.write(line)

    shutil.rmtree('/home/grf/PycharmProjects/cmmrt/resources/Classification_files/')

if os.path.exists('/home/grf/PycharmProjects/cmmrt/resources/all_classified_sorted.tsv'):
    print("Zipping")
    # Zip the generated files
    with zipfile.ZipFile('/home/grf/PycharmProjects/cmmrt/resources/50Mcompounds_sorted.zip', 'w', zipfile.ZIP_DEFLATED) as file_to_zip:
        file_to_zip.write('/home/grf/PycharmProjects/cmmrt/resources/all_classified_sorted.tsv', arcname='all_classified_sorted.tsv')
    os.remove('/home/grf/PycharmProjects/cmmrt/resources/all_classified_sorted.tsv')
