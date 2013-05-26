/*
 * FileUtils.cpp
 *
 *  Created on: May 20, 2013
 *      Author: matt
 */

#include "FileUtils.h"
#include <stdio.h>
#include <dirent.h>


FileUtils::FileUtils() {
}

FileUtils::~FileUtils() {
}



void FileUtils::findFilesInDir(string dir, vector<FileInfo*>& files)
{
	DIR *dp;
	struct dirent *dirp;

	if((dp  = opendir(dir.c_str())) == NULL) {
		cout << "Error opening " << dir << endl;
	}

	while ((dirp = readdir(dp)) != NULL)
	{
		string file = string(dirp->d_name);

		FileInfo* pFileInfo = new FileInfo();
		pFileInfo->fileName = file;
		pFileInfo->path = dir;

		files.push_back(pFileInfo);

		if((dirp->d_type & DT_DIR) && file.compare(".") != 0 && file.compare("..") != 0)
		{
			findFilesInDir(dir+"/"+dirp->d_name, files);
		}


	}
	closedir(dp);
}

void FileUtils::replace(std::string& str, const std::string& from, const std::string& to)
{
	while(_replace(str, from, to))
	{
	}
}
bool FileUtils::_replace(std::string& str, const std::string& from, const std::string& to)
{
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}
