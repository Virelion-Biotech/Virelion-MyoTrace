# Public input acquisition and repeatability recovery

The failed hosted public-input job reported a checksum mismatch for `20kPa.tif`.
An independent fresh download of all three pinned movies matched every original
SHA-256 and byte size, including the 102,497,938-byte 20kPa source. No source pin,
event detector, numerical threshold or biological interpretation was changed.
The historical failure's exact transient cause is not established by this probe.

Acquisition now rehashes cached bytes, downloads to a temporary sibling, checks
both declared size and SHA-256, and installs only verified bytes atomically.
Interrupted/network transfers have bounded retries. Permanent HTTP and integrity
errors remain errors; a failed replacement preserves existing bytes and removes
the partial download. Source failures receive a structured failure artifact.

The complete local public-video runner executed all 12 movie/method combinations
and verified exact trace repeatability on a second analysis of each input.
`public_execution.zip` contains `public/results.json` with source receipts, QC, summaries and descriptive
method disagreements; the traces/events are retained in the same archive. Raw source
movies are not redistributed. All 101 software tests pass; seven targeted
acquisition checks cover valid/corrupt caches, truncation, same-size corruption,
interrupted transfers and permanent HTTP failures. Changed-file Ruff passes.

The final helper additionally catches HTTP IncompleteRead; that failure branch
was added during the public-input run and is covered by its regression test.
The public run encountered no transfer errors and reused checksum-verified bytes.
These inputs do not supply independently annotated beat/force truth, so neither
repeatability nor method agreement establishes biological timing or force accuracy.
