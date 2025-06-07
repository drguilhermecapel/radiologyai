; MedAI Radiologia - Instalador Windows
; Instalador NSIS para sistema de análise radiológica com IA

!define APPNAME "MedAI Radiologia"
!define COMPANYNAME "MedAI Systems"
!define DESCRIPTION "Sistema de Análise Radiológica com Inteligência Artificial"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://github.com/drguilhermecapel/radiologyai"
!define UPDATEURL "https://github.com/drguilhermecapel/radiologyai/releases"
!define ABOUTURL "https://github.com/drguilhermecapel/radiologyai"
!define INSTALLSIZE 2048000

RequestExecutionLevel admin
InstallDir "$PROGRAMFILES\${APPNAME}"
LicenseData "LICENSE.txt"
Name "${APPNAME}"
Icon "medai_icon.ico"
outFile "MedAI_Radiologia_Installer.exe"

!include LogicLib.nsh

page license
page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${If} $0 != "admin"
    messageBox mb_iconstop "Privilégios de administrador são necessários para instalar o ${APPNAME}."
    setErrorLevel 740
    quit
${EndIf}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    
    ; Arquivos principais
    file "dist\MedAI_Radiologia.exe"
    file /r "dist\*"
    
    ; Criar diretórios necessários
    createDirectory "$INSTDIR\data"
    createDirectory "$INSTDIR\models"
    createDirectory "$INSTDIR\reports"
    createDirectory "$INSTDIR\temp"
    
    ; Arquivos de configuração
    file /oname=config.json "medai_config_default.json"
    
    ; Documentação
    createDirectory "$INSTDIR\docs"
    file /oname=docs\Manual_Usuario.pdf "..\docs\USER_GUIDE.md"
    file /oname=docs\Instalacao.pdf "..\docs\INSTALLATION.md"
    
    ; Atalhos no Menu Iniciar
    createDirectory "$SMPROGRAMS\${APPNAME}"
    createShortCut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\MedAI_Radiologia.exe" "" "$INSTDIR\MedAI_Radiologia.exe"
    createShortCut "$SMPROGRAMS\${APPNAME}\Manual do Usuário.lnk" "$INSTDIR\docs\Manual_Usuario.pdf"
    createShortCut "$SMPROGRAMS\${APPNAME}\Desinstalar.lnk" "$INSTDIR\uninstall.exe"
    
    ; Atalho na Área de Trabalho
    createShortCut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\MedAI_Radiologia.exe" "" "$INSTDIR\MedAI_Radiologia.exe"
    
    ; Registro no Windows
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayName" "${APPNAME} - ${DESCRIPTION}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "QuietUninstallString" "$\"$INSTDIR\uninstall.exe$\" /S"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "InstallLocation" "$\"$INSTDIR$\""
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayIcon" "$\"$INSTDIR\MedAI_Radiologia.exe$\""
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "Publisher" "${COMPANYNAME}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "HelpLink" "${HELPURL}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLUpdateInfo" "${UPDATEURL}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "URLInfoAbout" "${ABOUTURL}"
    writeRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "DisplayVersion" "${VERSIONMAJOR}.${VERSIONMINOR}.${VERSIONBUILD}"
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMajor" ${VERSIONMAJOR}
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "VersionMinor" ${VERSIONMINOR}
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoModify" 1
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "NoRepair" 1
    writeRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" "EstimatedSize" ${INSTALLSIZE}
    
    ; Associações de arquivo para DICOM
    writeRegStr HKCR ".dcm" "" "MedAI.DICOM"
    writeRegStr HKCR "MedAI.DICOM" "" "Arquivo DICOM - MedAI Radiologia"
    writeRegStr HKCR "MedAI.DICOM\DefaultIcon" "" "$INSTDIR\MedAI_Radiologia.exe,0"
    writeRegStr HKCR "MedAI.DICOM\shell\open\command" "" "$\"$INSTDIR\MedAI_Radiologia.exe$\" $\"%1$\""
    
    ; Criar desinstalador
    writeUninstaller "$INSTDIR\uninstall.exe"
    
    ; Mensagem de sucesso
    messageBox MB_OK "Instalação concluída com sucesso!$\r$\n$\r$\nO ${APPNAME} foi instalado em:$\r$\n$INSTDIR$\r$\n$\r$\nVocê pode executar o programa através do atalho na área de trabalho ou no menu iniciar."
    
sectionEnd

section "uninstall"
    ; Remover arquivos
    delete "$INSTDIR\MedAI_Radiologia.exe"
    delete "$INSTDIR\uninstall.exe"
    rmDir /r "$INSTDIR\dist"
    rmDir /r "$INSTDIR\docs"
    rmDir /r "$INSTDIR\temp"
    
    ; Manter dados do usuário (opcional)
    messageBox MB_YESNO "Deseja manter os dados e configurações do usuário?" IDYES KeepData
    rmDir /r "$INSTDIR\data"
    rmDir /r "$INSTDIR\models"
    rmDir /r "$INSTDIR\reports"
    delete "$INSTDIR\config.json"
    
    KeepData:
    rmDir "$INSTDIR"
    
    ; Remover atalhos
    delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
    delete "$SMPROGRAMS\${APPNAME}\Manual do Usuário.lnk"
    delete "$SMPROGRAMS\${APPNAME}\Desinstalar.lnk"
    rmDir "$SMPROGRAMS\${APPNAME}"
    delete "$DESKTOP\${APPNAME}.lnk"
    
    ; Remover entradas do registro
    deleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"
    deleteRegKey HKCR ".dcm"
    deleteRegKey HKCR "MedAI.DICOM"
    
sectionEnd
