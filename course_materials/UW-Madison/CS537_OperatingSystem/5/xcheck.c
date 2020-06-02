#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <limits.h>
#include <string.h>

#define stat xv6_stat
#define dirent xv6_dirent
#include "types.h"
#include "fs.h"
#include "stat.h"
#undef stat
#undef dirent

const char* USAGE_ERR = "Usage: xcheck <file_system_image>"; // ok
const char* IMG_NOT_FOUND_ERR = "image not found."; // ok

const char* ERR_BAD_INODE = "ERROR: bad inode."; // 1: ok
const char* ERR_BAD_DIRECT_ADDR = "ERROR: bad direct address in inode."; // 2.1: ok
const char* ERR_BAD_INDIRECT_ADDR = "ERROR: bad indirect address in inode."; // 2.2: ok
const char* ERR_ROOT = "ERROR: root directory does not exist."; // 3: ok
const char* ERR_BAD_DIR_FORMAT = "ERROR: directory not properly formatted."; // 4: ok

const char* ERR_USED_ADDR_BM_ZERO = "ERROR: address used by inode but marked free in bitmap."; // 5: ok
const char* ERR_UNUSED_ADDR_BM_ONE = "ERROR: bitmap marks block in use but it is not in use."; // 6: ok

////  use bitmap
const char* ERR_DIRECT_ADDR_USED_MULTIPLE = "ERROR: direct address used more than once."; // 7: ok
const char* ERR_INDIRECT_ADDR_USED_MULTIPLE = "ERROR: indirect address used more than once."; // 8: not tested
////

const char* ERR_USED_INODE_NOT_IN_DIR = "ERROR: inode marked use but not found in a directory."; // 9: ok
const char* ERR_UNUSED_INODE_IN_DIR = "ERROR: inode referred to in directory but marked free."; // 10: ok

const char* ERR_BAD_REF_COUNT_FOR_FILE = "ERROR: bad reference count for file."; // 11: ok
const char* ERR_DIR_APPEAR_MULTIPLE = "ERROR: directory appears more than once in file system."; // 12: ok

void raise_error(const char* err_msg)
{
    fprintf(stderr, "%s\n", err_msg);
    exit(1);
}

// ToDo: modify
char* itoa_base2(uint value, char* buffer)
{ 
	// invalid input
	// if (base < 2 || base > 32) return buffer;
	// consider absolute value of number
    int base = 2;
	uint n = value;

	int i = 0;
	while (n) {
		int r = n % base;
		if (r>=10) buffer[i++] = 65 + (r-10);
		else buffer[i++] = 48 + r;
		n = n / base;
	}

	if (i == 0) buffer[i++] = '0';
	if (value < 0 && base == 10) buffer[i++] = '-';

	while (i < 32)
		buffer[i++] = '0';

	buffer[i] = '\0'; // null terminate string

	// reverse the string and return it
	return buffer;
}

int main(int argc, char* argv[])
{
    int fd;
    if (argc == 2){
        fd = open(argv[1], O_RDONLY);
    } else {
        raise_error(USAGE_ERR);   
    }
    if (fd < 0){
        raise_error(IMG_NOT_FOUND_ERR);
    }

    struct stat sbuf;
    fstat(fd, &sbuf);    
    // printf ("Image I read is %ld in size\n", sbuf.st_size);

    // mmap
    void* img_ptr = mmap(NULL, sbuf.st_size,\
                        PROT_READ, MAP_PRIVATE, fd, 0);
    if (img_ptr == -1){
        // printf("mmap fail");
        exit(1);
    }
    // fclose(fd);

    // | unused | super | inodes
    struct superblock *sb = (struct superblock*)(img_ptr + BSIZE);
    // printf("%d inodes\n", sb->ninodes);

    // inode table
    //     inode number for / is 1
    //     inode table starts from block 2

    struct dinode* inodes = (struct dinode*) (img_ptr + 2*BSIZE);
    uint block_addr;
    struct xv6_dirent* directories;
    uint* indirect_addrs;
    
    int block_used_amounts[sb->size];
    for (int i = 0; i < sb->size; i++)  block_used_amounts[i] = 0;

    if (inodes[1].type != T_DIR)
        raise_error(ERR_ROOT);

    // A. Enumerate Inodes
    for (int it_inode = 1; it_inode < sb->ninodes; it_inode++){
        // legal type: 0, T_FILE, T_DIR, T_DEV
        if (inodes[it_inode].type == 0)
            continue;
        if (inodes[it_inode].type < 0 || T_DEV < inodes[it_inode].type){
            raise_error(ERR_BAD_INODE);
        }
        // check T_DIR related
        if (inodes[it_inode].type == T_DIR){
            block_addr = inodes[it_inode].addrs[0];
            directories = (struct xv6_dirent*)(img_ptr + block_addr*BSIZE);
            if (strcmp(directories[0].name, ".") != 0 || strcmp(directories[1].name, "..") != 0 ||\
                directories[0].inum != it_inode)
                raise_error(ERR_BAD_DIR_FORMAT);
            if (it_inode == 1 && (directories[0].inum != 1 || directories[1].inum != 1))
                raise_error(ERR_ROOT);
        }
        // Enumerate Direct Addr
        for (int it_addr = 0; it_addr < NDIRECT; it_addr++){
            block_addr = inodes[it_inode].addrs[it_addr];
            if (block_addr == 0)
                continue;
            if (block_addr < sb->size-sb->nblocks || sb->size <= block_addr) // 29, 1024
                raise_error(ERR_BAD_DIRECT_ADDR);
            block_used_amounts[block_addr]++;
            if (block_used_amounts[block_addr] > 1)
                raise_error(ERR_DIRECT_ADDR_USED_MULTIPLE);
        }
        // Indirect Addr
        if (inodes[it_inode].addrs[NDIRECT] == 0) // check if the final block be in use
            continue;
        block_used_amounts[inodes[it_inode].addrs[NDIRECT]]++;
        if (block_used_amounts[inodes[it_inode].addrs[NDIRECT]] > 1)
            raise_error(ERR_DIRECT_ADDR_USED_MULTIPLE);// still check for direct addr

        indirect_addrs = (uint*)(img_ptr + BSIZE*inodes[it_inode].addrs[NDIRECT]);
        int it_indirect_addr = 0;
        while (it_indirect_addr < BSIZE/sizeof(uint) &&\
                indirect_addrs[it_indirect_addr] != 0 ){
            if (indirect_addrs[it_indirect_addr] < sb->size-sb->nblocks ||\
                sb->size <= indirect_addrs[it_indirect_addr]) // 29, 1024
                raise_error(ERR_BAD_INDIRECT_ADDR);
            block_used_amounts[indirect_addrs[it_indirect_addr]]++;
            if (block_used_amounts[indirect_addrs[it_indirect_addr]] > 1)
                raise_error(ERR_INDIRECT_ADDR_USED_MULTIPLE);
            it_indirect_addr++;
        }
        // 
    }

    // bitmap
    char* bitmap = malloc(sizeof(char) * 1024);
    uint* bitptr = (uint*) (img_ptr + 28 * BSIZE);
    // so we need 1024/ 8 = 128 bytes, and 128 bytes = 32 uints
    for (int i=0; i<32; i++){
        char buffer[4];
        strcat(bitmap, itoa_base2(bitptr[i], buffer));
    }

    for (int it_addr = 29; it_addr < sb->size; it_addr++){
        if (block_used_amounts[it_addr] == 1 && bitmap[it_addr] == '0')
            raise_error(ERR_USED_ADDR_BM_ZERO);
        if (block_used_amounts[it_addr] == 0 && bitmap[it_addr] == '1')
            raise_error(ERR_UNUSED_ADDR_BM_ONE);
    }
    free(bitmap);

    // check inode used by directory
    int inode_used_amounts[sb->ninodes];
    for (int i = 0; i < sb->ninodes; i++)  inode_used_amounts[i] = 0;
    
    for (int it_inode = 1; it_inode < sb->ninodes; it_inode++){
        if (inodes[it_inode].type != T_DIR)
            continue;
        // Enumerate Direct Addr
        for (int it_addr = 0; it_addr < NDIRECT; it_addr++){
            block_addr = inodes[it_inode].addrs[it_addr];
            if (block_addr == 0)
                continue;
            directories = (struct xv6_dirent*) (img_ptr + block_addr*BSIZE);
            for (int it_directory = 0; it_directory < BSIZE/sizeof(struct xv6_dirent); it_directory++){
                if (strcmp(directories[it_directory].name, ".") == 0 || strcmp(directories[it_directory].name, "..") == 0)
                    continue;
                if (directories[it_directory].inum > 0)
                    inode_used_amounts[directories[it_directory].inum]++;
            }
        }
        // Indirect Addr
        indirect_addrs = (uint*)(img_ptr + BSIZE*inodes[it_inode].addrs[NDIRECT]);
        // int it_indirect_addr = 0;
        // while (it_indirect_addr < BSIZE/sizeof(uint) &&\
        //         indirect_addrs[it_indirect_addr] != 0 ){
        //     directories = (struct xv6_dirent*) (img_ptr + indirect_addrs[it_indirect_addr]*BSIZE);
        //     for (int it_directory = 0; it_directory < BSIZE/sizeof(struct xv6_dirent); it_directory++){
        //         if (directories[it_directory].inum > 0)
        //             inode_used_amounts[directories[it_directory].inum]++;
        //     }
        //     it_indirect_addr++;
        // }
        for (int it_indirect_addr = 0; it_indirect_addr < BSIZE/sizeof(uint); it_indirect_addr++){
            if (indirect_addrs[it_indirect_addr] == 0)
                continue;
            directories = (struct xv6_dirent*) (img_ptr + indirect_addrs[it_indirect_addr]*BSIZE);
            for (int it_directory = 0; it_directory < BSIZE/sizeof(struct xv6_dirent); it_directory++){
                if (strcmp(directories[it_directory].name, ".") == 0 || strcmp(directories[it_directory].name, "..") == 0)
                    continue;
                if (directories[it_directory].inum > 0)
                    inode_used_amounts[directories[it_directory].inum]++;
            }
        }
    }
    inode_used_amounts[1] = 1;

    for (int it_inode = 1; it_inode < sb->ninodes; it_inode++){
        if (inodes[it_inode].type != 0 && inode_used_amounts[it_inode] == 0)
            raise_error(ERR_USED_INODE_NOT_IN_DIR);
        if (inodes[it_inode].type == 0 && inode_used_amounts[it_inode] != 0)
            raise_error(ERR_UNUSED_INODE_IN_DIR);
        if (inodes[it_inode].type == T_FILE && inodes[it_inode].nlink != inode_used_amounts[it_inode])
            raise_error(ERR_BAD_REF_COUNT_FOR_FILE);
        if (inodes[it_inode].type == T_DIR && inode_used_amounts[it_inode] > 1)
            raise_error(ERR_DIR_APPEAR_MULTIPLE);    
    }
    
    // for (int it_inode = 1; it_inode < sb->ninodes; it_inode++)
    //     if (inodes[it_inode].type == T_FILE && inodes[it_inode].nlink != inode_used_amounts[it_inode])
    //         raise_error(ERR_BAD_REF_COUNT_FOR_FILE);

    // for (int it_inode = 1; it_inode < sb->ninodes; it_inode++){
    //     if (inodes[it_inode].type == T_DIR && inode_used_amounts[it_inode] > 1)
    //         raise_error(ERR_DIR_APPEAR_MULTIPLE);
    // }

    // fclose(fd); why can not operate this?
    exit(0);
}