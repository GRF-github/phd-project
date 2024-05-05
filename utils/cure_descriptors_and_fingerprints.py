import pandas


def cure_metlin(descriptors, fingerprints):
    # Drop NaNs
    fingerprints.dropna(subset=['Molecule Name'], inplace=True)
    fingerprints.drop(columns=['dimer line', 'CCS', 'm/z.2', 'pubChem', 'METLIN ID',], inplace=True)
    fingerprints.drop(fingerprints[fingerprints['Molecule Name'].str.contains("Tm_")].index, inplace=True)

    descriptors.dropna(subset=['Molecule Name'], inplace=True)
    descriptors.drop(columns=['dimer line', 'CCS', 'm/z.2', 'pubChem', 'METLIN ID',], inplace=True)
    descriptors.drop(descriptors[descriptors['Molecule Name'].str.contains("Tm_")].index, inplace=True)


    def calculate_average_ccs(row):
        """
        Calculate the average Collision Cross Section (CCS) value from three different experiments.

        Parameters:
        - row (pandas.Series): A row from a DataFrame containing columns 'CCS1', 'CCS2', and 'CCS3' with CCS values.

        Returns:
        - float: The average CCS value rounded to two decimal places.
        """

        # Extract values from columns CCS1, CCS2, CCS3 for the current row
        ccs1 = row['CCS1']
        ccs2 = row['CCS2']
        ccs3 = row['CCS3']

        # Calculate the average and round to two decimals
        average_ccs = round((ccs1 + ccs2 + ccs3) / 3, 2)

        return average_ccs

    fingerprints['correct_ccs_avg'] = fingerprints.apply(calculate_average_ccs, axis=1)
    descriptors['correct_ccs_avg'] = descriptors.apply(calculate_average_ccs, axis=1)

    # Identify each compound with a unique number
    fingerprints.reset_index(inplace=True)
    descriptors.reset_index(inplace=True)

    # Use pandas.get_dummies() to create binary columns
    adduct_dummies = pandas.get_dummies(fingerprints['Adduct'])

    # Concatenate the original DataFrame with the new binary columns
    fingerprints = pandas.concat([fingerprints, adduct_dummies], axis=1)

    # Remove bloat columns
    columns_to_drop = ['Molecule Name', 'Molecular Formula', 'Precursor Adduct', 'CCS1', 'CCS2', 'CCS3', 'CCS_AVG',
                       '% CV', 'm/z', 'Adduct', 'm/z.1', 'Dimer', 'Dimer.1', 'inchi', 'smiles', 'InChIKEY']
    fingerprints.drop(columns=columns_to_drop, inplace=True)
    descriptors.drop(columns=columns_to_drop, inplace=True)

    # Return cured descriptors and cured fingerprints
    return descriptors, fingerprints


def cure_allccs2(descriptors, fingerprints):
    # Parameters:
    keep_only_same_adducts_as_in_Metlin = True

    # Remove any compound without InChI or Adduct
    fingerprints.dropna(subset=['InChI', 'Adduct'], inplace=True)
    descriptors.dropna(subset=['InChI', 'Adduct'], inplace=True)

    # Filter the dataframe in we only want to have the same adduct as in Metlin
    if keep_only_same_adducts_as_in_Metlin:
        desired_adducts = ['[M+H]+', '[M-H]-', '[M+Na]+']
        mask_of_desired_adducts = fingerprints['Adduct'].isin(desired_adducts)
        rows_to_drop = fingerprints[~mask_of_desired_adducts].index
        fingerprints.drop(rows_to_drop, inplace=True)
        descriptors.drop(rows_to_drop, inplace=True)

    # Identify each compound with a unique number
    fingerprints.reset_index(inplace=True)
    descriptors.reset_index(inplace=True)

    # Create the one-hot encoding columns from the column 'Adduct' and concatenate them with the rest of the dataframe
    adduct_dummies = pandas.get_dummies(fingerprints['Adduct'])
    fingerprints = pandas.concat([fingerprints, adduct_dummies], axis=1)

    # Remove non-used columns
    columns_to_drop = ['AllCCS ID', 'Name', 'Structure', 'Formula', 'Type', 'm/z', 'Confidence level', 'Update date',
                       'InChI', 'Adduct']
    fingerprints.drop(columns=columns_to_drop, inplace=True)
    descriptors.drop(columns=columns_to_drop, inplace=True)

    # Return cured descriptors and cured fingerprints
    return descriptors, fingerprints
