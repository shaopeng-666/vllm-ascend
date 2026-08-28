# Source reconstruction

> PR 版本不包含 `codex-c8-full-0065f5f3.tar.gz`：该压缩快照在 Git 上传时返回 HTTP 403，且内容可由本目录的 bundle/patch 恢复。下面“方式一”只适用于另行保存的原始离线包；从本 PR 接力时请使用“方式二”或后面的 patch。

## 方式一：直接恢复精确 tracked 树

```bash
tar -xzf codex-c8-full-0065f5f3.tar.gz
cd codex-c8-full
```

该归档对应：

```text
0065f5f3bd88c1608932273dd9543d1919a55bbb
debug(mla): gate C8 cache host readback
```

归档不含 `.git` 历史，也不含 submodule 内容。submodule 记录为：

```text
csrc/third_party/catlass
41bf90da655bba3c66d0acd7e00abe33960ecfd6
https://gitcode.com/cann/catlass.git
```

## 方式二：从 Git bundle 恢复历史

需要先让本地 Git 拥有两个公开前置 commit：

```text
23cfbd5bae8028dc31d8ddd66673f985cf66e5f5
d37a76b431378096538c1818cb92fba51c5801f0
```

建议从公开仓库获取，再依次导入 bundle：

```bash
git init codex-c8-history
cd codex-c8-history
git fetch https://github.com/vllm-project/vllm-ascend.git \
  23cfbd5bae8028dc31d8ddd66673f985cf66e5f5 \
  d37a76b431378096538c1818cb92fba51c5801f0

git fetch ../codex-c8-full-f7dde85c8.bundle \
  HEAD:refs/heads/codex-full-f7
git fetch ../codex-c8-full-60462f479.bundle \
  HEAD:refs/heads/codex-full-604
git fetch ../codex-c8-capture-guard.bundle \
  HEAD:refs/heads/codex-capture-guard
git fetch ../codex-c8-host-readback-gate.bundle \
  HEAD:refs/heads/codex-host-readback

git checkout -b recovered-c8-probe \
  0065f5f3bd88c1608932273dd9543d1919a55bbb
git submodule update --init --recursive
```

若托管端拒绝按未广告 SHA fetch，可先 clone 公开仓库，再 fetch 包含这两个 commit 的公开分支/PR ref；也可以先使用 tar.gz 精确恢复 tracked 树，不依赖 Git 历史。

## Patch

`codex-c8-full-d37a76b-to-0065f5f3.patch` 是从 merge 前 checkout `d37a76b...` 到最终恢复 commit `0065f5f3...` 的 `git diff --binary`。应用前应先确认 base：

```bash
test "$(git rev-parse HEAD)" = \
  d37a76b431378096538c1818cb92fba51c5801f0
git apply --check ../codex-c8-full-d37a76b-to-0065f5f3.patch
git apply ../codex-c8-full-d37a76b-to-0065f5f3.patch
```

该 patch 可恢复文件内容，但不会重建 merge/commit 历史。

## 完整性限制

此处“精确”仅指 Git tracked tree。旧服务器断连前没有保存的 dirty/untracked 文件无法从 bundle 推导，详见 `UNTRACKED-STATUS.txt`。

四个 bundle 只导出了 `HEAD`，没有保留旧工作区分支名；本文中的 `codex-full-f7`、`codex-capture-guard` 等是恢复时创建的描述性 ref，不是对旧 branch 名的声明。公开前置源码来自 `https://github.com/vllm-project/vllm-ascend.git`，但旧工作区当时的精确 `git remote -v` 当前不可恢复。
