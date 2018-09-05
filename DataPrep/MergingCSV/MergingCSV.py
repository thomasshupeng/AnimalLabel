# -*- coding: utf-8 -*-
"""
Syntax:
    python MergingCSV.py
@author: Shu Peng
"""

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

import os
import pandas as pd
import chardet
from shutil import copyfile

cwd = os.getcwd()

root = Tk()
APP_TITLE = "MergingCSV v1.1"

source_path = StringVar()
source_path.set(cwd)

output_file = StringVar()
output_file.set(os.path.join(cwd, "Merged.csv"))


def file_encoding(filename):
    firstline = None
    encoding = "utf-8"
    with open(filename, 'rb') as f:
        firstline = f.readline()
    if firstline:
        encoding = chardet.detect(firstline)['encoding']
    return encoding


def browse_path(parent_window):
    init_dir = os.getcwd()
    dlg_title = "Select root folder of all CSV files"
    init_dir = source_path.get()
    pathname = filedialog.askdirectory(parent=parent_window, title=dlg_title, initialdir=init_dir)
    if pathname:
        pathname=pathname.replace('/', '\\')
        source_path.set(pathname)
    return


def browse_outputfile(parent_window):
    dlg_title = "Select output file"
    init_dir, init_file = os.path.split(output_file.get())

    filename = filedialog.asksaveasfilename(parent=parent_window, filetypes = [("CSV File","*.csv")],
                                            title=dlg_title, initialdir = init_dir, initialfile=init_file)
    if filename:
        filename, extname = os.path.splitext(filename)
        filename = filename + ".csv"
        filename=filename.replace('/', '\\')
        output_file.set(filename)
    return


def merge_csv():
    source_dir = source_path.get()
    merged_file = output_file.get()
    if not os.path.exists(source_dir):
        messagebox.showerror("Path not found!")
        return
    if len(merged_file) == 0:
        messagebox.showerror("Error", "Output file name not specified")
        return
    else:
        merged_path, merged_filename = os.path.split(merged_file)
        if not os.path.exists(merged_path):
            if messagebox.askokcancel("Path not found", "Do you want to create the path? ("+merged_path+")"):
                os.makedirs(merged_path)
            else:
                return

    csv_file_list = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f[-4:].lower() == '.csv':
                csv_file_list.append(os.path.join(root, f))
    merged_df = pd.DataFrame()
    df_list = []
    try_gb18030 = False
    for csvfile in csv_file_list:
        try:
            temp_df = pd.read_csv(csvfile, encoding=file_encoding(csvfile))
            df_list.append(temp_df)
        
        except UnicodeDecodeError as e:
            try_gb18030 = True
        
        except :
            print("Error in reading csv file ", csvfile)
            print("Copying this file to NotMeraged folder")
            not_merged = os.path.join(merged_path, "NotMerged")
            path, filename = os.path.split(csvfile)
            if not os.path.exists(not_merged):
                os.makedirs(not_merged)
            copyfile(csvfile, os.path.join(not_merged, filename))
        
        if try_gb18030 :
            try_gb18030 = False
            try:
                temp_df = pd.read_csv(csvfile, encoding='gb18030')
                df_list.append(temp_df)
            except :
                print("Error in reading csv file ", csvfile)
                print("Copying this file to NotMeraged folder")
                not_merged = os.path.join(merged_path, "NotMerged")
                path, filename = os.path.split(csvfile)
                if not os.path.exists(not_merged):
                    os.makedirs(not_merged)
                    copyfile(csvfile, os.path.join(not_merged, filename))

    merged_df = pd.concat(df_list, axis=0, sort=False)
    print("Merged csv shape:")
    print(merged_df.shape)
    print(merged_df.dtypes)
    print("Removing dupliates")
    merged_df.drop_duplicates(inplace=True)
    print(merged_df.shape)
    merged_df.to_csv(merged_file, encoding='utf-8-sig', index=False)
    return


def main():
    main_frame = Frame(root)
    main_frame.master.title(APP_TITLE)
    main_frame.pack()

    path_lf = LabelFrame(main_frame, text="Root Path")
    path_lf.pack(side=TOP, padx=5, pady=5)
    source_path_entry = Entry(path_lf,textvariable=source_path, width=80)
    source_path_entry.grid(row=0, column=1, sticky=W, padx=5, pady=2)
    btn_browse_source = Button(path_lf, text="Browse ...", command=lambda: browse_path(main_frame))
    btn_browse_source.grid(row=0, column=2, sticky=W, padx=2, pady=2)

    output_lf = LabelFrame(main_frame, text="Save as")
    output_lf.pack(side=TOP, padx=5, pady=5)
    dest_path_entry = Entry(output_lf,textvariable=output_file, width=80)
    dest_path_entry.grid(row=1, column=1, sticky=W, padx=5, pady=2)
    btn_browse_destin = Button(output_lf, text="Browse ...", command=lambda: browse_outputfile(main_frame))
    btn_browse_destin.grid(row=1, column=2, sticky=W, padx=2, pady=2)

    cmd_frame = Frame(main_frame)
    cmd_frame.pack(side=TOP, padx=5, pady=5, fill=X)
    btn_merge = Button(cmd_frame, text="Merge", width=10, command=merge_csv)
    btn_merge.pack(side=LEFT, padx=80, pady=2)
    btn_exit = Button(cmd_frame, text="Exit", width=10, command=root.destroy)
    btn_exit.pack(side=RIGHT, padx=80, pady=2)

    root.mainloop()
    return

if __name__ == '__main__':
    main()
