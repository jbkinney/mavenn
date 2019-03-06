import pandas as pd
import os
import qc
import utils
from utils import ControlledError, check
from mpathic import SortSeqError
import sys
from Bio import SeqIO
from utils import clean_SortSeqError

#@clean_SortSeqError
class io_local_class:

    def __init__(self):

        self._input_check()


    def _input_check(self):
        pass

    def format_fasta(self, s):
        '''Function which takes in the raw fastq format and returns only the sequence'''
        # find end of first line(which shows when sequence actually starts
        location = s.find('\n')
        s = s[location + 1:]
        # remove all line terminators
        s = s.replace('\n', '')
        return s

    def load_text(self, file_arg):
        """
        General function used to load data from a text file
        """
        file_handle = self.validate_file_for_reading(file_arg)
        try:
            df = pd.io.parsers.read_csv(file_handle, delim_whitespace=True, \
                                        comment='#', skip_blank_lines=True, engine='c')
        except:
            raise SortSeqError( \
                'Could not interpret text file %s as dataframe.' % repr(file_handle))
        return df.dropna(axis=0, how='all')  # Drop rows with all NaNs


    def validate_file_for_reading(self, file_arg):
        """ Checks that a specified file exists and is readable. Returns a valid file handle given a file name or handle
        """
        # If user passed file name
        if type(file_arg) == str:

            # Verify that file exists
            if not os.path.isfile(file_arg):
                raise SortSeqError('Cannot find file: %s' % file_arg)

            # Verify that file can be read
            if not os.access(file_arg, os.R_OK):
                raise SortSeqError('Can find but cannot read from file: %s' % file_arg)

            # Get handle to file
            file_handle = open(file_arg, 'r')

        # If user passed file object
        elif type(file_arg) == file:

            # Verify that file isn't closed
            if file_arg.closed:
                raise SortSeqError('File object is already closed.')
            file_handle = file_arg

        # Otherwise, throw error
        else:
            raise SortSeqError('file_arg is neigher a name or handle.')

        # Return validated file handle
        return file_handle


    def validate_file_for_writing(self, file_arg):
        """ Checks that a specified file can be written
        """
        # If user passed file name
        if type(file_arg) == str:

            # Get handle to file
            file_handle = open(file_arg, 'w')

            # Verify that file can be read
            if not os.access(file_arg, os.W_OK):
                raise SortSeqError('Cannot write to file: %s' % file_arg)

        # If user passed file object
        elif type(file_arg) == file:

            file_handle = file_arg

        # Otherwise, throw error
        else:
            raise SortSeqError('file_arg is neigher a name or handle.')

        # Return validated file handle
        return file_handle


    # JBK: This function is currently too complex to be refactored into load()
    def load_dataset(self, file_arg, file_type='text', seq_type=None):
        """ Loads a dataset file, returns a data frame.
            Can load text, fasta, or fastq files
        """

        # Check that the file can be read
        file_handle = self.validate_file_for_reading(file_arg)

        # Check that file type is vaild
        if not file_type in ['text', 'fasta', 'fastq']:
            raise SortSeqError('Argument file_type, = %s, is not valid.' % \
                               str(file_type))

        # Make sure seq_type, if any, is valid
        if seq_type and not (seq_type in qc.seqtypes):
            raise SortSeqError('seq_type %s is invalid' % str(seq_type))

        # If seq_type is specified, get correposnding colname
        if file_type == 'fasta':
            if not seq_type:
                raise SortSeqError('Seqtype unspecified while fasta file.')

            # Set column name based on seq_type
            colname = qc.seqtype_to_seqcolname_dict[seq_type]

            # Fill in dataframe with fasta or fastq data
            # First read in the sequence and sequence identifier into each line
            df = pd.io.parsers.read_csv(file_arg, lineterminator='>', engine='c', names='s')
            df.rename(columns={'s': colname}, inplace=True)
            # now remove sequence identifiers
            df.loc[:, colname] = df.loc[:, colname].apply(self.format_fasta)


        # If type is fastq, set sequence type to dna
        elif file_type == 'fastq':
            if not seq_type:
                seq_type = 'dna'
            if seq_type != 'dna':
                raise SortSeqError( \
                    'seq_type=%s is incompatible with file_type="fastq"' % seq_type)

            # Set column name based on seq_type
            colname = qc.seqtype_to_seqcolname_dict[seq_type]

            # Fill in dataframe with fasta or fastq data
            temp_df = pd.io.parsers.read_csv(file_arg, engine='c', names='s')
            temp_df.rename(columns={'s': colname}, inplace=True)
            df = pd.DataFrame(columns=[colname])
            df.loc[:, colname] = temp_df.loc[1::4, colname]
            df.reset_index(inplace=True, drop=True)

        # For text file, just load as whitespace-delimited data frame
        elif file_type == 'text':

            df = self.load_text(file_arg)

            # If seq_type is specified, get corresponding column name
            if seq_type:
                colname = qc.seqtype_to_seqcolname_dict[seq_type]

            # If seq_type is specified, make sure it matches colname in df
            if seq_type and (not colname in df.columns):
                raise SortSeqError('Column %s is not in dataframe' % str(colname))

        # Otherwise, raise error
        else:
            raise SortSeqError('Unrecognized filetype %s' % file_type)

        # Make sure data was actually loaded
        if not df.shape[0] >= 1:
            SortSeqError('No data was loaded.')

        # If sequences or tags were loaded,
        if any([qc.is_col_type(col, ['seqs', 'tag']) for col in df.columns]):

            # If there are no counts, then add counts
            ct_cols = [col for col in df.columns if qc.is_col_type(col, 'cts')]
            if len(ct_cols) == 0:
                df['ct'] = 1

            # Otherwise, if there are counts but no 'ct' column
            elif not 'ct' in ct_cols:
                df['ct'] = df.loc[:, ct_cols].sum(axis=1)

        # Return validated/fixed dataframe
        return qc.validate_dataset(df, fix=True)


    def load_contigs_from_fasta(self, file_arg, model_df, chunksize=10000, circular=False):
        L = model_df.shape[0]
        contig_list = []

        inloc = self.validate_file_for_reading(file_arg)
        for i, record in enumerate(SeqIO.parse(inloc, 'fasta')):
            name = record.name if record.name else 'contig_%d' % i
            # Split contig up into chunk)size bits
            full_contig_str = str(record.seq)

            # Add a bit on end if circular
            if circular:
                full_contig_str += full_contig_str[:L - 1]

                # Define chunks containing chunksize sites
            start = 0
            end = start + chunksize + L - 1
            while end < len(full_contig_str):
                contig_str = full_contig_str[start:end]
                contig_list.append((contig_str, name, start))
                start += chunksize
                end = start + chunksize + L - 1
            contig_str = full_contig_str[start:]
            contig_list.append((contig_str, name, start))

        return contig_list


    # JBK: I want to get rid of these
    def load_model(self, file_arg):
        return self.load(file_arg, file_type='model')


    def load_filelist(self, file_arg):
        return self.load(file_arg, file_type='filelist')


    def load_tagkey(self, file_arg):
        return self.load(file_arg, file_type='tagkey')


    def load_profile_ct(self, file_arg):
        return self.load(file_arg, file_type='profile_ct')


    def load_profile_freq(self, file_arg):
        return self.load(file_arg, file_type='profile_freq')


    def load_profile_mut(self, file_arg):
        return self.load(file_arg, file_type='profile_mut')


    def load_profile_info(self, file_arg):
        return self.load(file_arg, file_type='profile_info')


    def load_meanstd(self, file_arg):
        return self.load(file_arg, file_type='meanstd')


    def load_sitelist(self, file_arg):
        return self.load(file_arg, file_type='sitelist')


    # JBK: I want to switch to using only this function
    def load(self, file_arg, file_type, **kwargs):
        """ Loads file of any specified type
        """
        validate_func_dict = {
            # 'dataset'       : qc.validate_dataset,  # This won't work right now
            'model': qc.validate_model,
            'filelist': qc.validate_filelist,
            'tagkey': qc.validate_tagkey,
            'profile_ct': qc.validate_profile_ct,
            'profile_freq': qc.validate_profile_freq,
            'profile_mut': qc.validate_profile_mut,
            'profile_info': qc.validate_profile_info,
            'meanstd': qc.validate_meanstd,
            'sitelist': qc.validate_sitelist
        }

        df = self.load_text(file_arg)

        if 'dataset' in file_type:
            raise SortSeqError('file_type %s is not supported in load()' % file_type)

        if file_type not in validate_func_dict.keys():
            raise SortSeqError('Unrecognized file_type %s' % file_type)

        func = validate_func_dict[file_type]
        return func(df, fix=True, **kwargs)


    def write(self, df, file_arg, fast=False):
        """ Writes a data frame to specified file, given as name or handle
        """
        file_handle = self.validate_file_for_writing(file_arg)
        if fast:
            df.to_csv(file_handle, sep='\t', float_format='%10.6f')
        else:
            pd.set_option('max_colwidth', int(1e6))  # Dont truncate columns
            df_string = df.to_string( \
                index=False, col_space=5, float_format=utils.format_string)
            file_handle.write(df_string + '\n')  # Add trailing return
            file_handle.close()

