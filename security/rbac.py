"""
Security reference data: RBAC matrix, security controls, and Bicep IaC template.

RBAC_MATRIX       → tuples of (role, read, write, delete, manage_idx, view_logs, network)
SECURITY_CONTROLS → tuples of (name, status, detail)
BICEP_IaC         → deployable Bicep template for Azure AI Search + OpenAI + Key Vault
"""

# (role, Read, Write, Delete, ManageIndex, ViewLogs, Network)
RBAC_MATRIX: list[tuple] = [
    ("Search Admin",  True,  True,  True,  True,  True,  True),
    ("Contributor",   True,  True,  False, False, False, True),
    ("Search Reader", True,  False, False, False, False, True),
    ("Indexer (MI)",  False, True,  False, True,  False, False),
    ("Auditor",       True,  False, False, False, True,  False),
    ("Anonymous",     False, False, False, False, False, False),
]

SECURITY_CONTROLS: list[tuple[str, str, str]] = [
    ("Transport Encryption", "ACTIVE", "TLS 1.3; HTTPS enforced; no plain HTTP"),
    ("Data at Rest (CMK)",   "ACTIVE", "AES-256 via customer-managed key in Key Vault"),
    ("Private Endpoints",    "ACTIVE", "Public network disabled; PE for Search + OpenAI"),
    ("Managed Identity",     "ACTIVE", "System-assigned MI; no stored credentials"),
    ("Secret Management",    "ACTIVE", "API keys in Key Vault; env-var injection only"),
]

BICEP_IaC = """\
// azure-rag-infra.bicep
// Run: az deployment group create -g <rg> -f azure-rag-infra.bicep -p env=prod

param env      string = 'prod'
param location string = resourceGroup().location
param subnetId string

resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name:     'rag-search-${env}'
  location: location
  sku:      { name: 'standard' }
  properties: {
    replicaCount:        2
    partitionCount:      1
    publicNetworkAccess: 'disabled'
    semanticSearch:      'free'
    encryptionWithCmk:   { enforcement: 'Enabled' }
    disableLocalAuth:    false
  }
  identity: { type: 'SystemAssigned' }
}

resource openAI 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name:     'rag-openai-${env}'
  location: location
  kind:     'OpenAI'
  sku:      { name: 'S0' }
  properties: {
    publicNetworkAccess: 'Disabled'
    customSubDomainName: 'rag-openai-${env}'
  }
  identity: { type: 'SystemAssigned' }
}

resource embedDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: openAI
  name:   'text-embedding-3-small'
  properties: {
    model: { format: 'OpenAI', name: 'text-embedding-3-small', version: '1' }
  }
  sku: { name: 'Standard', capacity: 120 }
}

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name:     'kv-rag-${env}'
  location: location
  properties: {
    sku:                     { family: 'A', name: 'standard' }
    tenantId:                tenant().tenantId
    enableSoftDelete:        true
    enablePurgeProtection:   true
    enableRbacAuthorization: true
  }
}

resource peSearch 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: 'pe-search-${env}'
  location: location
  properties: {
    subnet: { id: subnetId }
    privateLinkServiceConnections: [{
      name: 'plsc-search'
      properties: { privateLinkServiceId: searchService.id, groupIds: ['searchService'] }
    }]
  }
}

resource peOpenAI 'Microsoft.Network/privateEndpoints@2023-09-01' = {
  name: 'pe-openai-${env}'
  location: location
  properties: {
    subnet: { id: subnetId }
    privateLinkServiceConnections: [{
      name: 'plsc-openai'
      properties: { privateLinkServiceId: openAI.id, groupIds: ['account'] }
    }]
  }
}

resource kvRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: kv
  name:  guid(kv.id, searchService.id, 'kv-secrets-user')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '4633458b-17de-408a-b874-0445c86b69e6'
    )
    principalId:   searchService.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output searchEndpoint string = 'https://${searchService.name}.search.windows.net'
output openAIEndpoint  string = openAI.properties.endpoint
output kvUri           string = kv.properties.vaultUri"""
