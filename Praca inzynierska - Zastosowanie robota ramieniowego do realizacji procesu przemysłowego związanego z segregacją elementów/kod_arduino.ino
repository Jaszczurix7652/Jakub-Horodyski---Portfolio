#include <Wire.h>
#include <Adafruit_INA219.h>
#include <SPI.h>

using namespace std;
Adafruit_INA219 ina219;

 

void setup(void) 
{
  Serial.begin(115200);
  while (!Serial) {
      delay(1);
  }

  uint32_t currentFrequency;
    
  Serial.println("Hello!");
  
  if (! ina219.begin()) {
    Serial.println("Nie mozna wykryc czujnika");
    while (1) { delay(10); }
  }
  
  ina219.setCalibration_16V_400mA();

  Serial.println("Rozpoczynanie pomiaru...");
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(4, OUTPUT);
}


int i = 0;
void loop(void) 
{
  float shuntvoltage = 0;
  float busvoltage = 0;
  float current_mA = 0;
  float loadvoltage = 0;
  float power_mW = 0;
  float rezystancja = 0;
  float rezystywnosc = 0;
  float srednia_rezystywnosc = 0;
  float rezystywnosc_1 = 0;
  float rezystywnosc_2 = 0;
  float rezystywnosc_3 = 0;
  float rezystywnosc_4 = 0;
  float rezystywnosc_5 = 0;
  
  int liczba_elementow = 0;
  int liczba_miedz = 0;
  int liczba_aluminium = 0;
  int liczba_stal = 0;

  int k=0;

  float d = 0;
  float l = 0;
  float S = 0;
  
    digitalWrite(2, LOW);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);

  String msg="Podaj srednice elementu w metrach: ";
  String msg2 = "Srednica: ";
  String msg3= "Podaj dlugosc elementu w metrach: "; 
  String msg4= "Dlugosc: ";

  if (i==0){
  Serial.println (msg);delay(10000);
  while (Serial.available()==0){
  }
  d = Serial.parseFloat();
  
  }
  if(d!=0){
  Serial.print(msg2); Serial.print(d); Serial.println(" m");
  i++;
  
  }
  if (i==1){
  Serial.println (msg3);delay(10000);
  while (Serial.available()==0){

  }
  l = Serial.parseFloat();
  
  }
  if (l!=0){
  Serial.print(msg4); Serial.print(l); Serial.println(" m");

  i++;
  }
  
 
  S = (3.14*(pow(d,2))/4);

  if (i==2){

  obliczanie:

    digitalWrite(2, LOW);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);

  shuntvoltage = ina219.getShuntVoltage_mV();
  busvoltage = ina219.getBusVoltage_V();
  current_mA = ina219.getCurrent_mA();
  power_mW = ina219.getPower_mW();

  loadvoltage = busvoltage + (shuntvoltage / 1000);
  rezystancja = (loadvoltage/(-current_mA*0.001));
  rezystywnosc = ((rezystancja*S)/l);
  
  
  Serial.print("Bus Voltage:   "); Serial.print(busvoltage,4); Serial.println(" V");
  Serial.print("Shunt Voltage: "); Serial.print(shuntvoltage,4); Serial.println(" mV");
  Serial.print("Load Voltage:  "); Serial.print(loadvoltage,4); Serial.println(" V");
  Serial.print("Current:       "); Serial.print(current_mA,4); Serial.println(" mA");
  Serial.print("Power:         "); Serial.print(power_mW,4); Serial.println(" mW");
  Serial.print("Rezystancja:   "); Serial.print(rezystancja,6); Serial.println(" Ohm");
  Serial.print("Rezystywnosc:  "); Serial.print(rezystywnosc,8); Serial.println(" Ohm*m");
  Serial.println("");

  delay(5000);

  if(k==0 && rezystywnosc>=0.001 && rezystywnosc<=0.1){
    rezystywnosc_1=rezystywnosc;
    k++;
    srednia_rezystywnosc=0;
  }

  else if(k==1 && rezystywnosc>=0.001 && rezystywnosc<=0.1){
    rezystywnosc_2=rezystywnosc;
    k++;
  }

  else if(k==2 && rezystywnosc>=0.001 && rezystywnosc<=0.1){
    rezystywnosc_3=rezystywnosc;
    k++;
  }

  else if(k==3 && rezystywnosc>=0.001 && rezystywnosc<=0.1){
    rezystywnosc_4=rezystywnosc;
    k++;
  }

  else if(k==4 && rezystywnosc>=0.001 && rezystywnosc<=0.1){
    rezystywnosc_5=rezystywnosc;
    
  
    srednia_rezystywnosc = (rezystywnosc_1 + rezystywnosc_2 + rezystywnosc_3 + rezystywnosc_4 + rezystywnosc_5)/5;
    Serial.print("Srednia rezystywnosc:  "); Serial.print(srednia_rezystywnosc,8); Serial.println(" Ohm*m");
    k=0;
  }
  
  if (srednia_rezystywnosc>=0.0011 && srednia_rezystywnosc<=0.0016){
    digitalWrite(2, HIGH);
    digitalWrite(3, LOW);
    digitalWrite(4, LOW);

    liczba_elementow++;
    liczba_miedz++;
    Serial.print("Liczba elementow:   "); Serial.print(liczba_elementow); Serial.print('\n');
    Serial.print("Liczba miedzianych elementow:   "); Serial.print(liczba_miedz); Serial.print('\n');
    Serial.println("");
    delay(10000);
    
  } else if(srednia_rezystywnosc>= 0.0018 && srednia_rezystywnosc<=0.0035 ){
    digitalWrite(3, HIGH);
    digitalWrite(2, LOW);
    digitalWrite(4, LOW);

    liczba_elementow++;
    liczba_aluminium++;
    Serial.print("Liczba elementow:   "); Serial.print(liczba_elementow); Serial.print('\n');
    Serial.print("Liczba aluminiowych elementow:   "); Serial.print(liczba_aluminium); Serial.print('\n');
    Serial.println("");
    delay(10000);

  } else if(srednia_rezystywnosc>=0.005 && srednia_rezystywnosc<=0.06 ){
    digitalWrite(4, HIGH);
    digitalWrite(2, LOW);
    digitalWrite(3, LOW);

    liczba_elementow++;
    liczba_stal++;
    Serial.print("Liczba elementow:   "); Serial.print(liczba_elementow); Serial.print('\n');
    Serial.print("Liczba stalowych elementow:   "); Serial.print(liczba_stal); Serial.print('\n');
    Serial.println("");
    delay(10000);

  }

  srednia_rezystywnosc=0;
  goto obliczanie;
  }
}

