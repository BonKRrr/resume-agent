#include "llm/ProcUtil.hpp"

#include <windows.h>

namespace procutil {

std::string run_capture_stdout(const std::string& cmdline_utf8) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE readPipe = NULL;
    HANDLE writePipe = NULL;

    if (!CreatePipe(&readPipe, &writePipe, &sa, 0)) {
        return "";
    }

    // Ensure the read end is not inherited
    SetHandleInformation(readPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = writePipe;
    si.hStdError  = writePipe;

    PROCESS_INFORMATION pi{};
    std::string cmdline = cmdline_utf8; // CreateProcessA needs a mutable buffer

    BOOL ok = CreateProcessA(
        NULL,
        cmdline.data(),
        NULL,
        NULL,
        TRUE,
        CREATE_NO_WINDOW,
        NULL,
        NULL,
        &si,
        &pi
    );

    // Parent no longer needs write end
    CloseHandle(writePipe);

    if (!ok) {
        CloseHandle(readPipe);
        return "";
    }

    std::string out;
    out.reserve(8192);

    char buf[4096];
    DWORD n = 0;

    while (true) {
        BOOL r = ReadFile(readPipe, buf, sizeof(buf), &n, NULL);
        if (!r || n == 0) break;
        out.append(buf, buf + n);
    }

    CloseHandle(readPipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    return out;
}

} // namespace procutil
