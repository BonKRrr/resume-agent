SHELL := cmd.exe
.SHELLFLAGS := /C

VCVARS := "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

CXX := cl
CXXFLAGS := /std:c++20 /EHsc /Ithird_party\onnxruntime\include /Iinclude
LDFLAGS := /link /LIBPATH:third_party\onnxruntime\lib onnxruntime.lib
TARGET := resume-agent.exe

APP_SRC := app\main.cpp

CMD_SRC := \
	src\commands\resumeDump.cpp \
	src\commands\analyze.cpp \
	src\commands\embed.cpp \
	src\commands\build.cpp

RESUME_SRC := \
	src\resume\Scorer.cpp \
	src\resume\BulletScoresArtifact.cpp \
	src\resume\SemanticMatcher.cpp

EMB_SRC := \
	src\emb\WordPieceTokenizer.cpp \
	src\emb\MiniLmEmbedder.cpp

IO_SRC := \
	src\io\JsonIO.cpp

JOBS_SRC := \
	src\jobs\JobCorpus.cpp \
	src\jobs\TextUtil.cpp \
	src\jobs\TfidfSearch.cpp \
	src\jobs\EmbeddingIndex.cpp \
	src\jobs\RequirementExtractor.cpp

LLM_SRC := \
	src\llm\MockLLMClient.cpp \
	src\llm\OllamaLLMClient.cpp

SRCS := $(APP_SRC) $(CMD_SRC) $(RESUME_SRC) $(IO_SRC) $(JOBS_SRC) $(EMB_SRC) $(LLM_SRC)

all:
	call $(VCVARS) && $(CXX) $(CXXFLAGS) $(SRCS) /Fe:$(TARGET) $(LDFLAGS)

clean:
	del /Q *.obj $(TARGET) 2>NUL
