# Convert every .docx / .pptx in the given deliverable folders to PDF (Word / PowerPoint must be installed).
# Word fields (table of contents, page numbers) are updated before saving.
param([Parameter(Mandatory = $true)][string[]]$Folders)
$w = New-Object -ComObject Word.Application; $w.Visible = $false
$p = New-Object -ComObject PowerPoint.Application
foreach ($d in $Folders) {
    $d = (Resolve-Path $d).Path
    foreach ($f in Get-ChildItem $d -Filter *.docx | Where-Object { $_.Name -notlike '~$*' }) {
        $doc = $w.Documents.Open($f.FullName)
        foreach ($t in $doc.TablesOfContents) { $t.Update() }
        $doc.Fields.Update() | Out-Null
        $doc.Save()
        $doc.SaveAs([ref]($f.FullName -replace '\.docx$', '.pdf'), [ref]17)
        $doc.Close()
        Write-Output "pdf $($f.Name)"
    }
    foreach ($f in Get-ChildItem $d -Filter *.pptx | Where-Object { $_.Name -notlike '~$*' }) {
        $pr = $p.Presentations.Open($f.FullName, $true, $false, $false)
        $pr.SaveAs(($f.FullName -replace '\.pptx$', '.pdf'), 32)
        $pr.Close()
        Write-Output "pdf $($f.Name)"
    }
}
$w.Quit(); $p.Quit()
