/*
 * FileUtils.h
 *
 *  Created on: May 20, 2013
 *      Author: matt
 */

#ifndef FILEUTILS_H_
#define FILEUTILS_H_

#include <iostream>
#include <vector>

using namespace std;

class FileInfo
{
public:
	std::string path;
	std::string fileName;
};

class FileUtils {
public:
	FileUtils();
	virtual ~FileUtils();

	void findFilesInDir(std::string dir, vector<FileInfo*>& files);

	void replace(std::string& str, const std::string& from, const std::string& to);
	bool _replace(std::string& str, const std::string& from, const std::string& to);
};

#endif /* FILEUTILS_H_ */
